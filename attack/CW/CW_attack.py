"""
Based on CVPR'19: Generating 3D Adversarial Point Clouds.

https://github.com/code-roamer/AOF/blob/master/baselines/attack/CW/PerturbT.py
"""

import pdb
import time
import random

import torch
import torch.optim as optim
import numpy as np



def rand_row(array):
    row_total = array.shape[1]
    row_sequence = np.arange(row_total)
    np.random.shuffle(row_sequence)
    return array[:, row_sequence, :]

class CW:
    """Class for CW attack.
    """

    def __init__(self, model, trans_model, adv_func, clip_func, dist_func, attack_lr=1e-2,
                 init_weight=10., max_weight=80., binary_step=10, num_iter=500, attack_method="untarget"):
        """CW attack by perturbing points.
        Args:
            model (torch.nn.Module): victim model
            adv_func (function): adversarial loss function
            dist_func (function): distance metric
            attack_lr (float, optional): lr for optimization. Defaults to 1e-2.
            init_weight (float, optional): weight factor init. Defaults to 10.
            max_weight (float, optional): max weight factor. Defaults to 80.
            binary_step (int, optional): binary search step. Defaults to 10.
            num_iter (int, optional): max iter num in every search step. Defaults to 500.
        """

        self.model = model.cuda()
        self.model.eval()
        self.trans_model = trans_model.cuda()
        self.trans_model.eval()
        self.adv_func = adv_func
        self.dist_func = dist_func
        self.attack_lr = attack_lr
        self.init_weight = init_weight
        self.max_weight = max_weight
        self.binary_step = binary_step
        self.num_iter = num_iter
        self.clip_func = clip_func
        self.attack_method = attack_method
        self.shuffle_fail = 0

    def attack(self, data, target):
        """Attack on given data to target.
        Args:
            data (torch.FloatTensor): victim data, [B, num_points, 3]
            target (torch.LongTensor): target output, [B]
        """
        B, K = data.shape[:2]
        data = data.float().cuda().detach()
        data = data.transpose(1, 2).contiguous()
        ori_data = data.clone().detach()

        target = target.long().cuda().detach()
        label_val = target.detach().cpu().numpy()  # [B]

        # weight factor for budget regularization
        lower_bound = np.zeros((B,))
        upper_bound = np.ones((B,)) * self.max_weight
        current_weight = np.ones((B,)) * self.init_weight

        # record best results in binary search
        o_bestdist = np.array([1e10] * B)
        o_bestscore = np.array([-1] * B)
        o_bestattack = np.zeros((B, 3, K))

        adv_data = ori_data.clone().detach()
        logits, _, _ = self.model(adv_data)  # [B, num_classes]
        pred = torch.argmax(logits, dim=1)
        print("ori label:",pred.item())

        if self.attack_method == 'top1_error':
            pred_max1 = logits.topk(35, dim=1, largest=True, sorted=True)[1][0][30]
            target =label_val= pred_max1.unsqueeze(0)
            print(target.item())

        # perform binary search

        for binary_step in range(self.binary_step):
            adv_data = ori_data.clone().detach() + torch.randn((B, 3, K)).cuda() * 1e-7
            # init variables with small perturbation

            adv_data.requires_grad_()
            bestdist = np.array([1e10] * B)
            bestscore = np.array([-1] * B)
            opt = optim.Adam([adv_data], lr=self.attack_lr, weight_decay=0.)

            adv_loss = torch.tensor(0.).cuda()
            dist_loss = torch.tensor(0.).cuda()

            total_time = 0.
            forward_time = 0.
            backward_time = 0.
            update_time = 0.

            # one step in binary search
            for iteration in range(self.num_iter):
                t1 = time.time()

                # forward passing
                logits, _, _ = self.model(adv_data)

                t2 = time.time()
                forward_time += t2 - t1

                # print
                pred = torch.argmax(logits, dim=1)  # [B]

                if self.attack_method == "untarget":
                    success_num = (pred != target).sum().item()
                else:
                    success_num = (pred == target).sum().item()

                # record values!
                dist_val = torch.sqrt(torch.sum(
                    (adv_data - ori_data) ** 2, dim=[1, 2])). \
                    detach().cpu().numpy()  # [B]
                pred_val = pred.detach().cpu().numpy()  # [B]
                input_val = adv_data.detach().cpu().numpy()  # [B, 3, K]

                # update
                for e, (dist, pred, label, ii) in \
                        enumerate(zip(dist_val, pred_val, label_val, input_val)):
                    if self.attack_method == 'untarget':
                        if dist < bestdist[e] and pred != label:
                            bestdist[e] = dist
                            bestscore[e] = pred
                        if dist < o_bestdist[e] and pred != label:
                            o_bestdist[e] = dist
                            o_bestscore[e] = pred
                            o_bestattack[e] = ii
                    else:
                        if dist < bestdist[e] and pred == label:
                            bestdist[e] = dist
                            bestscore[e] = pred
                        if dist < o_bestdist[e] and pred == label:
                            o_bestdist[e] = dist
                            o_bestscore[e] = pred
                            o_bestattack[e] = ii


                t3 = time.time()
                update_time += t3 - t2

                # compute loss and backward
                adv_loss = self.adv_func(logits, target).mean()
                dist_loss = self.dist_func(adv_data, ori_data,
                                           torch.from_numpy(
                                               current_weight)).mean()

                loss = adv_loss + dist_loss
                opt.zero_grad()
                loss.backward()
                opt.step()

                # clipping and projection!
                if self.clip_func is not None:
                    adv_data.data = self.clip_func(adv_data.clone().detach(),
                                          ori_data)

                t4 = time.time()
                backward_time += t4 - t3
                total_time += t4 - t1


            # adjust weight factor
            for e, label in enumerate(label_val):
                if self.attack_method == 'untarget':
                    if bestscore[e] != label and bestscore[e] != -1 and bestdist[e] <= o_bestdist[e]:
                        # success
                        lower_bound[e] = max(lower_bound[e], current_weight[e])
                        current_weight[e] = (lower_bound[e] + upper_bound[e]) / 2.
                    else:
                        # failure
                        upper_bound[e] = min(upper_bound[e], current_weight[e])
                        current_weight[e] = (lower_bound[e] + upper_bound[e]) / 2.
                else:
                    if bestscore[e] == label and bestscore[e] != -1 and bestdist[e] <= o_bestdist[e]:
                        # success
                        lower_bound[e] = max(lower_bound[e], current_weight[e])
                        current_weight[e] = (lower_bound[e] + upper_bound[e]) / 2.
                    else:
                        # failure
                        upper_bound[e] = min(upper_bound[e], current_weight[e])
                        current_weight[e] = (lower_bound[e] + upper_bound[e]) / 2.


            torch.cuda.empty_cache()

        # end of CW attack
        # fail to attack some examples
        # just assign them with last time attack data
        fail_idx = (lower_bound == 0.)
        o_bestattack[fail_idx] = input_val[fail_idx]

        # Test attack
        attack_result = o_bestattack
        attack_result = torch.from_numpy(attack_result)
        attack_result = attack_result.float().cuda()
        attack_logits, _, _ = self.model(attack_result)
        print('attack result: ', torch.argmax(attack_logits, dim=1).item())

        # Test shuffle attack
        attack_result = o_bestattack.transpose((0, 2, 1))
        attack_result = rand_row(attack_result)
        attack_result = torch.from_numpy(attack_result.transpose((0, 2, 1)))
        attack_result = attack_result.float().cuda()
        shuffle_logits, _, _ = self.model(attack_result)
        print('shuffle result: ',torch.argmax(shuffle_logits, dim=1).item())
        if self.attack_method == 'untarget':
            if torch.argmax(shuffle_logits, dim=1) == target:
                self.shuffle_fail+=1
                print("shuffle fail: ", self.shuffle_fail)
        else:
            if torch.argmax(shuffle_logits, dim=1) != target:
                self.shuffle_fail += 1
                print("shuffle fail: ", self.shuffle_fail)

        # Test transfer attack
        transfer_result = o_bestattack
        transfer_result = torch.from_numpy(transfer_result)
        transfer_result = transfer_result.float().cuda()
        transfer_logits, _, _ = self.trans_model(transfer_result)
        print('transfer result: ', torch.argmax(transfer_logits, dim=1).item())

        # return final results
        # success_num = (lower_bound > 0.).sum()
        print('Successfully attack {}/{}'.format(success_num, B))
        if torch.argmax(transfer_logits, dim=1).item() != pred.item() and success_num==1:
            trans_success=1
        else:
            trans_success=0
        return o_bestdist, o_bestattack.transpose((0, 2, 1)), success_num, trans_success
