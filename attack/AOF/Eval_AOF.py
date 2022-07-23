import os
import pdb
import time
# from torch._C import _llvm_enabled
from tqdm import tqdm
import copy
import argparse

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR
from attack.CW.utils.dist_utils import ChamferDist
from model.curvenet import CurveNet
from model.pointnet import PointNetCls, feature_transform_regularizer
from model.pointnet2_MSG import PointNet_Msg
from model.pointnet2_SSG import PointNet_Ssg
from model.dgcnn import DGCNN
from torch.utils.data import DataLoader
from attack.CW.CW_utils.basic_util import cal_loss, AverageMeter, get_lr, str2bool, set_seed
from attack.CW.CW_utils.adv_utils import CrossEntropyAdvLoss, LogitsAdvLoss, UntargetedLogitsAdvLoss
from attack.CW.CW_utils.dist_utils import ClipPointsLinf, ChamferkNNDist
from dataset.bosphorus_dataset import Bosphorus_Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def pc_visualize(pc, file_name="visual/img.png"):
    data = pc.detach().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], marker='o')
    ax.view_init(elev=-90., azim=-90)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.savefig(file_name)
    plt.show()
    plt.close()


def knn(x, k):
    """
    x:(B, 3, N)
    """
    # print(f"input shape:{x.shape}")
    with torch.no_grad():
        inner = -2 * torch.matmul(x.transpose(2, 1), x)  # (B, N, N)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)  # (B, 1, N)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)  # (B, N, N)

        vec = x.transpose(2, 1).unsqueeze(2) - x.transpose(2, 1).unsqueeze(1)
        dist = -torch.sum(vec ** 2, dim=-1)
        # print("distance check:", torch.allclose(pairwise_distance, dist))

        # print(f"dist shape:{pairwise_distance.shape}")
        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N, k)
    return idx


def rand_row(array):
    row_total = array.shape[1]
    row_sequence = np.arange(row_total)
    np.random.shuffle(row_sequence)
    return array[:, row_sequence, :]


def get_Laplace_from_pc(ori_pc):
    """
    ori_pc:(B, 3, N)
    """
    # print("shape of ori pc:",ori_pc.shape)
    pc = ori_pc.detach().clone()
    with torch.no_grad():
        pc = pc.to('cpu').to(torch.double)
        idx = knn(pc, 30)
        pc = pc.transpose(2, 1).contiguous()  # (B, N, 3)
        point_mat = pc.unsqueeze(2) - pc.unsqueeze(1)  # (B, N, N, 3)
        A = torch.exp(-torch.sum(point_mat.square(), dim=3))  # (B, N, N)
        mask = torch.zeros_like(A)
        mask.scatter_(2, idx, 1)
        mask = mask + mask.transpose(2, 1)
        mask[mask > 1] = 1

        A = A * mask
        D = torch.diag_embed(torch.sum(A, dim=2))
        L = D - A
        e, v = torch.symeig(L, eigenvectors=True)
    return e.to(ori_pc), v.to(ori_pc)


def normalize_points(points):
    """points: [K, 3]"""
    points = points - torch.mean(points, 0, keepdim=True)  # center
    dist = torch.max(torch.sqrt(torch.sum(points ** 2, dim=1)))
    print(dist)
    points = points / dist  # scale

    return points


def need_clip(pc, ori_pc, budget=0.1):
    with torch.no_grad():
        diff = pc - ori_pc  # [B, 3, K]
        norm = torch.sum(diff ** 2, dim=1) ** 0.5  # [B, K]
        scale_factor = budget / (norm + 1e-9)  # [B, K]
        bt = scale_factor < 1.0
        bt = torch.sum(bt, dim=-1)
        mask = (bt > 0).to(torch.float)

    return mask


def attack():
    iter_num = 0
    at_num, total_num, trans_num = 0.0, 0.0, 0.0
    all_adv_pc = []
    all_real_lbl = []
    st = time.time()
    for data, label in tqdm(test_loader):
        iter_num += 1
        with torch.no_grad():
            data, label = data.transpose(2, 1).float().cuda(non_blocking=True), \
                          label.long().cuda(non_blocking=True)
        ori_data = data.detach().clone()
        ori_data.requires_grad_(False)

        B = data.shape[0]
        K = data.shape[2]
        # record best results in binary search
        o_bestdist = np.array([1e10] * B)
        o_bestscore = np.array([-1] * B)
        o_bestattack = np.zeros((B, 3, K))
        label_val = label.detach().cpu().numpy()  # [B]

        # perform binary search
        for binary_step in range(args.step):
            data = ori_data.clone().detach() + \
                   torch.randn((B, 3, K)).cuda() * 1e-7
            Evs, V = get_Laplace_from_pc(data)
            projs = torch.bmm(data, V)  # (B, 3, N)
            hfc = torch.bmm(projs[..., args.low_pass:], V[..., args.low_pass:].transpose(2, 1))  # (B, 3, N)
            lfc = torch.bmm(projs[..., :args.low_pass], V[..., :args.low_pass].transpose(2, 1))
            lfc = lfc.detach().clone()
            hfc = hfc.detach().clone()
            lfc.requires_grad_()
            hfc.requires_grad_(False)
            ori_lfc = lfc.detach().clone()
            ori_lfc.requires_grad_(False)
            ori_hfc = hfc.detach().clone()
            ori_hfc.requires_grad_(False)
            opt = optim.Adam([lfc], lr=args.lr,
                             weight_decay=0)
            # opt = optim.Adam([{'params': hfc, 'lr': 1e-2},
            #                     {'params': lfc, 'lr': 1e-2}], lr=args.lr,
            #              weight_decay=0)
            # attack training
            for i in range(args.epochs):
                adv_pc = lfc + hfc

                logits, _, _ = model(adv_pc)
                lfc_logits, _, _ = model(lfc)

                # record values!
                pred = torch.argmax(logits, dim=1)  # [B]
                lfc_pred = torch.argmax(lfc_logits, dim=1)  # [B]
                dist_val = torch.amax(torch.abs(
                    (adv_pc - data)), dim=(1, 2)). \
                    detach().cpu().numpy()  # [B]
                pred_val = pred.detach().cpu().numpy()  # [B]
                lfc_pred_val = lfc_pred.detach().cpu().numpy()  # [B]
                # print(pred_val, dist_val)

                input_val = adv_pc.detach().cpu().numpy()  # [B, 3, K]
                # update
                for e, (dist, pred, lfc_pred, label_e, ii) in \
                        enumerate(zip(dist_val, pred_val, lfc_pred_val, label_val, input_val)):
                    if pred != label_e and dist < o_bestdist[e] and lfc_pred != label_e:
                        o_bestdist[e] = dist
                        o_bestscore[e] = pred
                        o_bestattack[e] = ii

                loss = 0.5 * adv_func(logits, label) + 0.5 * adv_func(lfc_logits, label)
                opt.zero_grad()
                loss.backward()
                opt.step()

                # clip
                with torch.no_grad():
                    adv_pc = lfc + hfc
                    adv_pc.data = clip_func(adv_pc.detach().clone(), data)
                    coeff = torch.bmm(adv_pc, V)
                    hfc.data = torch.bmm(coeff[..., args.low_pass:],
                                         V[..., args.low_pass:].transpose(2, 1))  # (B, 3, N)
                    lfc.data = torch.bmm(coeff[..., :args.low_pass], V[..., :args.low_pass].transpose(2, 1))

            torch.cuda.empty_cache()

        # adv_pc = clip_func(adv_pc, data)
        # print("best linf distance:", o_bestdist)
        adv_pc = torch.tensor(o_bestattack).to(adv_pc)
        adv_pc = clip_func(adv_pc, data)

        logits, _, _ = model(adv_pc)
        preds = torch.argmax(logits, dim=-1)

        trans_logits, _, _ = trans_model(adv_pc)
        trans_preds = torch.argmax(trans_logits, dim=-1)

        shuffle_pc = adv_pc.transpose(2, 1).float()
        shuffle_pc = rand_row(shuffle_pc)
        shuffle_pc = shuffle_pc.transpose(2, 1).to(device)

        shuffle_logits, _, _ = model(shuffle_pc)
        shuffle_preds = torch.argmax(shuffle_logits, dim=-1)
        shuffle_trans_logits, _, _ = trans_model(shuffle_pc)
        shuffle_trans_preds = torch.argmax(shuffle_trans_logits, dim=-1)


        at_num += (preds != label).sum().item()
        trans_num += (trans_preds != label).sum().item()

        total_num += args.batch_size

        print("\n", preds)
        print(trans_preds)
        print(label)

        if iter_num % 1 == 0:
            print(f"attack success rate:{at_num / total_num}, trans success rate: {trans_num / total_num}")

        best_pc = adv_pc.transpose(1, 2).contiguous().detach().cpu().numpy()
        all_adv_pc.append(best_pc)
        all_real_lbl.append(label.detach().cpu().numpy())

    et = time.time()
    print(
        f"attack success rate:{at_num / total_num}, trans success rate: {trans_num / total_num}, consuming time:{et - st} seconds")
    all_adv_pc = np.concatenate(all_adv_pc, axis=0)  # [num_data, K, 3]
    all_real_lbl = np.concatenate(all_real_lbl, axis=0)  # [num_data]
    # save results
    """
    save_path = './attack/results/{}_{}/AOF/{}'. \
        format(args.dataset, args.num_point, args.model)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    success_rate = at_num / total_num
    save_name = 'AOF-{}-low_pass_{}-budget_{}-success_{:.4f}.npz'. \
        format(args.model, args.low_pass, args.budget,
               success_rate)
    np.savez(os.path.join(save_path, save_name),
             test_pc=all_adv_pc.astype(np.float32),
             test_label=all_real_lbl.astype(np.uint8))
    """
    # print(args)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--model', type=str, default='DGCNN', metavar='N',
                        help="Model to use, ['PointNet', 'PointNet++Msg','DGCNN', 'CurveNet']")
    parser.add_argument('--trans_model', type=str, default='PointNet', metavar='N',
                        help="Model to use, ['PointNet', 'PointNet++Msg','DGCNN', 'CurveNet']")
    parser.add_argument('--dataset', type=str, default='Bosphorus',
                        help='dataset : Bosphorus | Eurecom')
    parser.add_argument('--batch_size', type=int, default=1, metavar='BS',
                        help='Size of batch')
    parser.add_argument('--feature_transform', type=str2bool, default=True,
                        help='whether to use STN on features in PointNet')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train ')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate for the optimizer')
    parser.add_argument('--low_pass', type=int, default=100,
                        help='low_pass number')
    parser.add_argument('--budget', type=float, default=0.18,
                        help='FGM attack budget')
    parser.add_argument('--eig_budget', type=float, default=1.5,
                        help='FGM attack budget')
    parser.add_argument('--step', type=int, default=2, metavar='N',
                        help='Number of binary search step')
    parser.add_argument('--num_point', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--dropout', type=float, default=0.5, help='parameters in DGCNN: dropout rate')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings in DGCNN')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use in DGCNN')
    parser.add_argument('--adv_func', type=str, default='logits',
                        choices=['logits', 'cross_entropy'],
                        help='Adversarial loss function to use')
    parser.add_argument('--kappa', type=float, default=0.,
                        help='min margin in logits adv loss')
    parser.add_argument('--binary_step', type=int, default=1, metavar='N',
                        help='Binary search step')
    parser.add_argument('--num_iter', type=int, default=100, metavar='N',
                        help='Number of iterations in each search step')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--num_of_class', default=105 + 1, type=int,
                        help='number of class, 105+1  |  52')
    args = parser.parse_args()
    set_seed(1)

    # enable cudnn benchmark
    # cudnn.benchmark = True
    # BEST_WEIGHTS = BEST_WEIGHTS[args.dataset][args.num_point]

    # build victim model
    if args.model == 'PointNet':
        model = PointNetCls(k=args.num_of_class, feature_transform=False)
    elif args.model == 'PointNet++Msg':
        model = PointNet_Msg(args.num_of_class, normal_channel=False)
    elif args.model == 'PointNet++Ssg':
        model = PointNet_Ssg(args.num_of_class)
    elif args.model == 'DGCNN':
        model = DGCNN(args, output_channels=args.num_of_class).to(device)
    elif args.model == 'CurveNet':
        model = CurveNet(num_classes=args.num_of_class)
    else:
        exit('wrong model type')

    # model = nn.DataParallel(model).cuda()
    model.load_state_dict(
        torch.load(os.path.expanduser(
            '~//yq_pointnet//cls//{}//{}_model_on_{}.pth'.format(args.dataset, args.model, args.dataset))))
    model.eval()

    # build transfer model
    if args.trans_model == 'PointNet':
        trans_model = PointNetCls(k=args.num_of_class, feature_transform=False)
    elif args.trans_model == 'PointNet++Msg':
        trans_model = PointNet_Msg(args.num_of_class, normal_channel=False)
    elif args.trans_model == 'PointNet++Ssg':
        trans_model = PointNet_Ssg(args.num_of_class)
    elif args.trans_model == 'DGCNN':
        trans_model = DGCNN(args, output_channels=args.num_of_class).to(device)
    elif args.trans_model == 'CurveNet':
        trans_model = CurveNet(num_classes=args.num_of_class)
    else:
        exit('wrong model type')
    # trans_model = nn.DataParallel(trans_model).cuda()
    trans_model.load_state_dict(
        torch.load(os.path.expanduser(
            '~//yq_pointnet//cls//{}//{}_model_on_{}.pth'.format(args.dataset, args.trans_model, args.dataset))))
    trans_model.eval()

    # prepare dataset
    # test_set = ModelNetDataLoader(root=args.data_root, args=args, split='test', process_data=args.process_data)
    # test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=10, drop_last=False)

    test_dataset_path = os.path.expanduser("~//yq_pointnet//BosphorusDB//eval.csv")
    test_set = Bosphorus_Dataset(test_dataset_path)
    model.to(device)
    trans_model.to(device)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=True,
        num_workers=0)
    # test_set = CustomModelNet40('custom_data', num_points=args.num_point, normalize=True)
    # test_loader = DataLoader(test_set, batch_size=args.batch_size,
    #                          shuffle=False, num_workers=8,
    #                          pin_memory=True, drop_last=False)

    # test_set = ModelNet40Attack(args.attack_data_root, num_points=args.num_point,
    #                             normalize=True)
    # test_loader = DataLoader(test_set, batch_size=args.batch_size,
    #                          shuffle=False, num_workers=4,
    #                          pin_memory=True, drop_last=False)



    clip_func = ClipPointsLinf(budget=args.budget)
    adv_func = UntargetedLogitsAdvLoss(kappa=30.)
    dist_func = ChamferDist()

    attack()
