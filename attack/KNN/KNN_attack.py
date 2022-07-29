import pdb
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def rand_row(array, dim_needed):

    row_total = array.shape[0]
    row_sequence = np.arange(row_total)
    np.random.shuffle(row_sequence)
    return array[row_sequence[0:dim_needed], :]

class CWKNN:
    """Class for CW attack.
    """

    def __init__(self, model, pt_model,ptm_model,pts_model,dgcnn_model,adv_func, dist_func, clip_func,
                 attack_lr=1e-3, num_iter=2500,attack_method='untarget'):

        self.model = model.cuda()
        self.model.eval()


        self.pt_model = pt_model.cuda()
        self.pt_model.eval()

        self.ptm_model = ptm_model.cuda()
        self.ptm_model.eval()

        self.pts_model = pts_model.cuda()
        self.pts_model.eval()

        self.dgcnn_model = dgcnn_model.cuda()
        self.dgcnn_model.eval()

        self.adv_func = adv_func
        self.dist_func = dist_func
        self.clip_func = clip_func
        self.attack_lr = attack_lr
        self.num_iter = num_iter
        self.attack_method = attack_method
        self.shuffle_fail = 0
        self.trans_fail = 0
        self.attack_fail = 0
        self.pt_fail = 0
        self.ptm_fail = 0
        self.pts_fail = 0
        self.dgcnn_fail = 0

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
        ori_data.requires_grad = False

        # points and normals
        if ori_data.shape[1] == 3:
            normal = ori_data
        else:
            normal = ori_data[:, 3:, :]
            ori_data = ori_data[:, :3, :]


        logits, _, _ = self.model(ori_data)  # [B, num_classes]
        pred = torch.argmax(logits, dim=1)
        print("ori label:", pred.item())


        target = target.long().cuda().detach()

        # init variables with small perturbation
        adv_data = ori_data.clone().detach() + \
            torch.randn((B, 3, K)).cuda() * 1e-7
        adv_data.requires_grad_()
        opt = optim.Adam([adv_data], lr=self.attack_lr, weight_decay=0.)

        adv_loss = torch.tensor(0.).cuda()
        dist_loss = torch.tensor(0.).cuda()

        total_time = 0.
        forward_time = 0.
        backward_time = 0.
        clip_time = 0.

        # there is no binary search in this attack
        # just longer iterations of optimization
        for iteration in range(self.num_iter):
            t1 = time.time()

            # forward passing
            logits = self.model(adv_data)  # [B, num_classes]
            if isinstance(logits, tuple):  # PointNet
                logits = logits[0]

            t2 = time.time()
            forward_time += t2 - t1

            # print
            pred = torch.argmax(logits, dim=1)  # [B]
            if self.attack_method == 'untarget':
                success_num = (pred != target).sum().item()
            else:
                success_num = (pred == target).sum().item()
            # compute loss and backward
            adv_loss = self.adv_func(logits, target).mean()

            # in the official tensorflow code, they use sum instead of mean
            # so we multiply num_points as sum
            dist_loss = self.dist_func(
                adv_data.transpose(1, 2).contiguous(),
                ori_data.transpose(1, 2).contiguous()).mean() * K

            loss = adv_loss + dist_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            t3 = time.time()
            backward_time += t3 - t2

            # clipping and projection!

            adv_data.data = self.clip_func(adv_data.clone().detach(),ori_data,normal)


            t4 = time.time()
            clip_time = t4 - t3
            total_time += t4 - t1


        # end of CW attack
        with torch.no_grad():
            logits = self.model(adv_data)  # [B, num_classes]
            if isinstance(logits, tuple):  # PointNet
                logits = logits[0]
            pred = torch.argmax(logits, dim=-1)  # [B]
            if self.attack_method == 'untarget':
                success_num = (pred != target). \
                    sum().detach().cpu().item()
            else:
                success_num = (pred == target).\
                    sum().detach().cpu().item()

        # return final results
        print('Successfully attack {}/{}'.format(success_num, B))

        # Test attack
        attack_result = adv_data
        attack_result = attack_result.float().cuda()
        attack_logits, _, _ = self.model(attack_result)
        print('attack result: ', torch.argmax(attack_logits, dim=1).item())
        if self.attack_method == 'untarget':
            if torch.argmax(attack_logits, dim=1) == target:
                self.attack_fail += 1
                print("attack fail: ", self.attack_fail)
        else:
            if torch.argmax(attack_logits, dim=1) != target:
                self.attack_fail += 1
                print("attack fail: ", self.attack_fail)


        # Test transfer attack
        transfer_result = adv_data
        transfer_result = transfer_result.float().cuda()
        transfer_logits, _, _ = self.pt_model(transfer_result)
        print('pointnet result: ', torch.argmax(transfer_logits, dim=1).item())
        if self.attack_method == 'untarget':
            if torch.argmax(transfer_logits, dim=1) == target:
                self.pt_fail+=1
                print("pointnet fail: ", self.pt_fail)
        else:
            if torch.argmax(transfer_logits, dim=1) != target:
                self.pt_fail += 1
                print("pointnet fail: ", self.pt_fail)

        transfer_result = adv_data
        transfer_result = transfer_result.float().cuda()
        transfer_logits, _, _ = self.ptm_model(transfer_result)
        print('pointnet++msg result: ', torch.argmax(transfer_logits, dim=1).item())
        if self.attack_method == 'untarget':
            if torch.argmax(transfer_logits, dim=1) == target:
                self.ptm_fail += 1
                print("pointnet++msg fail: ", self.ptm_fail)
        else:
            if torch.argmax(transfer_logits, dim=1) != target:
                self.ptm_fail += 1
                print("pointnet++msg fail: ", self.ptm_fail)


        transfer_result = adv_data
        transfer_result = transfer_result.float().cuda()
        transfer_logits, _, _ = self.pts_model(transfer_result)
        print('pointnet++ssg result: ', torch.argmax(transfer_logits, dim=1).item())
        if self.attack_method == 'untarget':
            if torch.argmax(transfer_logits, dim=1) == target:
                self.pts_fail += 1
                print("pointnet++ssg fail: ", self.pts_fail)
        else:
            if torch.argmax(transfer_logits, dim=1) != target:
                self.pts_fail += 1
                print("pointnet++ssg fail: ", self.pts_fail)

        transfer_result = adv_data
        transfer_result = transfer_result.float().cuda()
        transfer_logits, _, _ = self.dgcnn_model(transfer_result)
        print('dgcnn result: ', torch.argmax(transfer_logits, dim=1).item())
        if self.attack_method == 'untarget':
            if torch.argmax(transfer_logits, dim=1) == target:
                self.dgcnn_fail += 1
                print("dgcnn fail: ", self.dgcnn_fail)
        else:
            if torch.argmax(transfer_logits, dim=1) != target:
                self.dgcnn_fail += 1
                print("dgcnn fail: ", self.dgcnn_fail)
        # in their implementation, they estimate the normal of adv_pc
        # we don't do so here because it's useless in our task
        adv_data = adv_data.transpose(1, 2).contiguous()  # [B, K, 3]
        adv_data = adv_data.detach().cpu().numpy()  # [B, K, 3]
        return adv_data, success_num
