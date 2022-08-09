import pdb
import time
import random
import torch
import torch.optim as optim
import numpy as np


# random.seed(7122)

class CW:
    """Class for CW attack.
    """

    def __init__(self, model, adv_func, dist_func, attack_lr=1e-2,
                 init_weight=10., max_weight=80., binary_step=10, num_iter=500, whether_target=True, whether_1d=True,
                 whether_renormalization=False,
                 whether_3Dtransform=False, whether_resample=False):
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

        self.adv_func = adv_func
        self.dist_func = dist_func
        self.attack_lr = attack_lr
        self.init_weight = init_weight
        self.max_weight = max_weight
        self.binary_step = binary_step
        self.num_iter = num_iter
        self.whether_target = whether_target
        self.whether_1d = whether_1d
        self.whether_renormalization = whether_renormalization
        self.whether_3Dtransform = whether_3Dtransform
        self.box_constraint = 0.4
        self.whether_resample = whether_resample

    def attack(self, data, target=torch.Tensor([0]), origin_label=torch.Tensor([105])):
        """Attack on given data to target.
        Args:
            data (torch.FloatTensor): victim data, [B, num_points, 3]
            target (torch.LongTensor): target output, [B]
        """
        if data.shape[2] == 3:
            data = data.transpose(1, 2).contiguous()  # [1,3,4000]
        K = data.shape[2]
        B = data.shape[0]
        data = data.float().cuda().detach()

        ori_data = data.clone().detach()
        # ori_data.requires_grad = False
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
        # if isinstance(logits, tuple):  # PointNet
        #    logits = logits[0]
        pred = torch.argmax(logits, dim=1)
        origin_label = origin_label.detach().cuda()
        # pred2 = logits.topk(15, dim=1, largest=True, sorted=True)[1][0][1]
        # print(logits.topk(15, dim=1, largest=True, sorted=True)[1][0])
        # target = torch.from_numpy(np.array(pred2.cpu())).cuda().unsqueeze(0)
        if self.whether_target:
            print('ori classify: {}  target:{}'.format(pred.item(), target.item()))
        else:
            print('ori classify: {}, clean pred:{} '.format(origin_label.item(), pred.item()))

        # perform binary search
        adv_data = ori_data.clone().detach() + torch.randn((B, 3, K)).cuda() * 1e-7
        for binary_step in range(self.binary_step):
            # init variables with small perturbation
            # adv_data = ori_data.clone().detach() + torch.randn((B, 3, K)).cuda() * 1e-9
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
                if self.whether_renormalization:
                    # print("renormalization")
                    adv_data_2 = adv_data.permute(0, 2, 1)
                    centroid = torch.mean(adv_data_2, axis=1)
                    adv_data3 = adv_data_2 - centroid.unsqueeze(1)
                    var = torch.max(torch.sqrt(torch.sum(adv_data3 ** 2, axis=2)), axis=1, keepdim=True)[0]
                    adv_data4 = adv_data3 / var.unsqueeze(1)
                    adv_data5 = adv_data4.permute(0, 2, 1)
                    logits, _, _ = self.model(adv_data5)
                else:
                    logits, _, _ = self.model(adv_data)
                # logits_set,_,_ = self.model(data_set)
                # if isinstance(logits, tuple):  # PointNet
                #     logits = logits[0]
                t2 = time.time()
                forward_time += t2 - t1
                # print
                pred = torch.argmax(logits, dim=1)  # [B]
                # print(logits.topk(15, dim=1, largest=True, sorted=True)[1][0])
                # if target
                if self.whether_target:
                    if pred == target:
                        success_num = 1
                    else:
                        success_num = 0
                # if untarget
                else:
                    if pred != origin_label:
                        success_num = 1
                    else:
                        success_num = 0
                # success_num = (pred != ori_label).sum().item() + success_num
                if iteration % 10 == 0:
                    print('Step {}, iteration {}, success {}/{}\n'
                          'adv_loss: {:.4f}, dist_loss: {:.4f}, current weight: {:.4f}'.
                          format(binary_step, iteration, success_num, B,
                                 adv_loss.item(), dist_loss.item(), current_weight.item()))

                # record values!

                dist_loss = self.dist_func(adv_data.permute(0, 2, 1), ori_data.permute(0, 2, 1),
                                           torch.from_numpy(
                                               current_weight))

                dist_val = dist_loss.detach().cpu().numpy()  # [B]
                pred_val = pred.detach().cpu().numpy()  # [B]
                input_val = adv_data.detach().cpu().numpy()  # [B, 3, K]

                dist_loss = dist_loss.mean()
                # update
                if self.whether_target:
                    print(dist_val, pred_val, label_val)
                    for e, (dist, pred, label, ii) in \
                            enumerate(zip([dist_val], pred_val, label_val, input_val)):
                        if dist < bestdist[e] and pred == target:
                            bestdist[e] = dist
                            bestscore[e] = pred
                        if dist < o_bestdist[e] and pred == target:
                            o_bestdist[e] = dist
                            o_bestscore[e] = pred
                            o_bestattack[e] = ii
                else:
                    for e, (dist, pred, label, ii) in \
                            enumerate(zip(dist_val, pred_val, label_val, input_val)):
                        if dist < bestdist[e] and pred != origin_label:
                            bestdist[e] = dist
                            bestscore[e] = pred
                        if dist < o_bestdist[e] and pred != origin_label:
                            o_bestdist[e] = dist
                            o_bestscore[e] = pred
                            o_bestattack[e] = ii

                t3 = time.time()
                update_time += t3 - t2

                # compute loss and backward

                # if target

                if self.whether_3Dtransform:
                    diff = adv_data - ori_data.detach()
                    adv_loss_list = torch.empty(10)
                    for i in range(10):
                        # rotation
                        theta = torch.randn(1) * 1e-2
                        Tz = torch.tensor([[torch.cos(theta), torch.sin(theta), 0],
                                           [-torch.sin(theta), torch.cos(theta), 0],
                                           [0, 0, 1]], dtype=torch.float32).unsqueeze(0).cuda()
                        Ty = torch.tensor([[torch.cos(theta), 0, torch.sin(theta)],
                                           [0, 1, 0],
                                           [-torch.sin(theta), 0, torch.cos(theta)]], dtype=torch.float32).unsqueeze(
                            0).cuda()
                        Tx = torch.tensor([[1, 0, 0],
                                           [0, torch.cos(theta), torch.sin(theta)],
                                           [0, -torch.sin(theta), torch.cos(theta)]], dtype=torch.float32).unsqueeze(
                            0).cuda()
                        To = torch.tensor([[1, 0, 0],
                                           [0, 1, 0],
                                           [0, 0, 1]], dtype=torch.float32).unsqueeze(0).cuda()
                        r = random.random()
                        if r < 0.2:
                            Tr = Tz
                        elif r < 0.4:
                            Tr = Tx
                        elif r < 0.6:
                            Tr = Ty
                        else:
                            Tr = To
                        # normal distribution noises
                        # zero = torch.randn(((B, 2, K))) * 1e-4 # x and y axis
                        # rand = torch.randn(((B, 1, K))) * 1e-2 # z axis

                        # adv_data2 =  torch.bmm(Tz.detach(), ori_data.detach()) + diff + torch.cat((zero,rand),1).cuda()
                        adv_data2 = torch.bmm(Tr.detach(), ori_data.detach()) + diff

                        # renormalization
                        if self.whether_renormalization:
                            adv_data_temp = adv_data2.permute(0, 2, 1)
                            centroid = torch.mean(adv_data_temp, axis=1)
                            adv_data3 = adv_data_temp - centroid.unsqueeze(1)
                            var = torch.max(torch.sqrt(torch.sum(adv_data3 ** 2, axis=2)), axis=1, keepdim=True)[0]
                            adv_data4 = adv_data3 / var.unsqueeze(1)
                            adv_data5 = adv_data4.permute(0, 2, 1)
                        else:
                            adv_data5 = adv_data2

                        if self.whether_resample:
                            adv_data6 = torch.cat((adv_data5, adv_data5), 2).cuda()
                            indices = random.sample(range(1, 4000 * 2), 4000)
                            adv_data7 = torch.index_select(adv_data6, 2, torch.LongTensor(indices).cuda()).cuda()
                        else:
                            adv_data7 = adv_data5

                        logits, _, _ = self.model(adv_data7)
                        # print("pred as: ", logits.topk(1, dim=1, largest=True, sorted=True)[1][0][0])
                        if self.whether_target:
                            adv_loss_list[i] = self.adv_func(logits, target, whether_target=1).mean()
                        else:
                            adv_loss_list[i] = self.adv_func(logits, origin_label, whether_target=0).mean()
                        adv_loss = torch.mean(adv_loss_list)
                    loss = adv_loss + dist_loss

                else:
                    if self.whether_target:
                        adv_loss = self.adv_func(logits, target, whether_target=1).mean()
                        loss = adv_loss + dist_loss

                    # if untarget
                    else:
                        adv_loss = self.adv_func(logits, origin_label, whether_target=0).mean()
                        loss = adv_loss + dist_loss

                opt.zero_grad()
                loss.backward()
                opt.step()

                # attack on z direction
                if self.whether_1d:
                    adv_data.requires_grad = False
                    adv_data[0, 0] = ori_data[0, 0].clone().detach()
                    adv_data[0, 1] = ori_data[0, 1].clone().detach()
                    # box constraint
                    adv_data[0, 2] = torch.max(torch.min(adv_data[0, 2], ori_data[0, 2] + self.box_constraint),
                                               ori_data[0, 2] - self.box_constraint)
                    adv_data.requires_grad = True

                # if iteration % 100 == 0:
                # total_time = 0.
                # forward_time = 0.
                # backward_time = 0.
                # update_time = 0.
            # adjust weight factor
            for e, label in enumerate(label_val):
                if self.whether_target == False:
                    if bestscore[e] != origin_label and bestscore[e] != -1 and bestdist[e] <= o_bestdist[e]:
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

        # return final results
        success_num = (lower_bound > 0.).sum()
        print('Successfully attack {}/{}   pred: {}'.format(success_num, B, pred))
        return o_bestdist, o_bestattack.transpose((0, 2, 1)), success_num







