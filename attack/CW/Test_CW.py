"""Targeted point perturbation attack."""

import os
import random

from tqdm import tqdm
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

import sys
random.seed(7122)
sys.path.append('../')
from dataset.bosphorus_dataset import Bosphorus_Dataset
# from config import BEST_WEIGHTS
# from config import MAX_PERTURB_BATCH as BATCH_SIZE
# from dataset import ModelNet40Attack

from attack.CW.CW_utils.basic_util import str2bool, set_seed
from attack.CW.CW_attack import CW
from attack.CW.CW_utils.adv_utils import CrossEntropyAdvLoss, LogitsAdvLoss
from attack.CW.CW_utils.dist_utils import L2Dist

from model.pointnet import PointNetCls, feature_transform_regularizer
from model.pointnet2_MSG import PointNet_Msg
from model.pointnet2_SSG import PointNet_Ssg
from model.dgcnn import DGCNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


point_cloud_txt_file_name = 'face0424smile1.txt'


def check_num_pc_changed(adv, ori):
    logits_mtx = np.logical_and.reduce(adv == ori, axis=1)
    return np.sum(logits_mtx == False)

def rand_row(array, dim_needed):
    row_total = array.shape[0]
    row_sequence = np.arange(row_total)
    np.random.shuffle(row_sequence)
    return array[row_sequence[0:dim_needed], :]

def attack(attacker):
    model.eval()
    all_adv_pc = []
    all_real_lbl = []
    all_target_lbl = []
    num = 0
    #for i, data in tqdm(enumerate(test_loader, 0)):
    test_data_path = os.path.expanduser("~//yq_pointnet//test_face_data/" + point_cloud_txt_file_name)

    ipt_ori = np.loadtxt(test_data_path, delimiter=',')
    #ipt, mu, st = normalization(ipt)
    #ipt_ori = rand_row(ipt_ori, 4793)
    #ipt_0 = []
    #for i in range(0, 90000, 20):
    #    ipt_0.append(ipt_ori[i])
    #print(len(ipt_0))
    ipt = ipt_ori[:, 0:3]
    ipt_c4c5 = ipt_ori[:, 3:5]
    #ipt = np.append(ipt, ipt, axis=0)
    mean = np.expand_dims(np.mean(ipt, axis=0), 0)
    ipt = ipt - mean
    var = np.max(np.sqrt(np.sum(ipt ** 2, axis=1)), 0)
    ipt = ipt / var  # scale
    print(ipt[0:10])
    # print((ipt*var+mean)[100:120])
    ipt = np.expand_dims(ipt, 0)
    ipt = torch.from_numpy(ipt).float()
    #ipt = ipt.permute(0, 2, 1)
    ipt = ipt.to(device)
    ipt.requires_grad_(True)
    pc, label = ipt, torch.tensor([105], dtype=torch.float).to(device)
    # target = label
    # target = random.randint(0, 105)
    flag = 0
    acount = 53
    while flag != 1:
        alist = []
        for j in range(len(label)):
            target = acount
            alist.append(target)
        acount  = acount + 1
        target = torch.tensor(alist)
        # target = target[:, 0]
        # pc = pc.transpose(2, 1)
        pc, target_label = pc.to(device='cuda', dtype=torch.float), target.cuda().float()
        print(pc.size())
        # attack!
        _, best_pc, success_num = attacker.attack(pc, target_label)

        # data_root = os.path.expanduser("~//yq_pointnet//attack/CW/AdvData/PointNet")
        # adv_f = '{}-{}-{}.txt'.format(i, int(label.detach().cpu().numpy()), int(target_label.detach().cpu().numpy()))

        # adv_fname = os.path.join(data_root, adv_f)
        if success_num > 0:
            print(best_pc.squeeze(0)[0:10])
            best_pc = best_pc.squeeze(0)*var + mean
            #best_pc_part1 = best_pc[0:4793]
            #result = np.concatenate((best_pc_part1, part2), axis=0)
            result = best_pc
            result = np.concatenate((result, ipt_c4c5), axis=1)
            # print(result[100:120])
            data_root = os.path.expanduser("~//yq_pointnet//test_face_data/")
            adv_f = 'adv_' + point_cloud_txt_file_name
            adv_fname = os.path.join(data_root, adv_f)
            #print(best_pc)
            #np.savetxt(adv_fname, best_pc, fmt='%.04f')
            np.savetxt(adv_fname, result, fmt='%.04f')

            res = best_pc

            target_pro = np.float32(np.squeeze(ipt.cpu().detach().numpy()))
            # Hausdorff_dis2 = dun.bid_hausdorff_dis(res, target_pro)
            # cham_dis = dun.chamfer(res, target_pro)
            num_preturbed_pc = check_num_pc_changed(res, target_pro)
            print('Finding advserial example Successful!')
            # print('Hausdorff distance: ', "%e" % Hausdorff_dis2)
            # print('Chamfer distance: ', "%e" % cham_dis)
            print('Number of points changed: ', num_preturbed_pc)
            flag = 1
        else:
            flag = 0

        # results
    num += success_num
    all_adv_pc.append(best_pc)
    all_real_lbl.append(label.detach().cpu().numpy())
    all_target_lbl.append(target_label.detach().cpu().numpy())

    # accumulate results
    all_adv_pc = np.concatenate(all_adv_pc, axis=0)  # [num_data, K, 3]
    all_real_lbl = np.concatenate(all_real_lbl, axis=0)  # [num_data]
    all_target_lbl = np.concatenate(all_target_lbl, axis=0)  # [num_data]
    return all_adv_pc, all_real_lbl, all_target_lbl, num


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--data_root', type=str,
                        default='data/attack_data.npz')
    parser.add_argument('--model', type=str, default='DGCNN', metavar='N',
                        choices=['PointNet', 'PointNet2_MSG', 'PointNet2_SSG',
                                 'DGCNN', 'pointconv'],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pointconv]')
    parser.add_argument('--feature_transform', type=str2bool, default=False,
                        help='whether to use STN on features in PointNet')
    parser.add_argument(
        '--dataset', type=str, default='Bosphorus', help="dataset: Bosphorus | Eurecom")
    parser.add_argument('--dropout', type=float, default=0.5, help='parameters in DGCNN: dropout rate')
    parser.add_argument('--batch_size', type=int, default=-1, metavar='BS',
                        help='Size of batch')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings in DGCNN')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use in DGCNN')
    parser.add_argument('--adv_func', type=str, default='cross_entropy',
                        choices=['logits', 'cross_entropy'],
                        help='Adversarial loss function to use')
    parser.add_argument('--kappa', type=float, default=0.,
                        help='min margin in logits adv loss')
    parser.add_argument('--attack_lr', type=float, default=5e-1,
                        help='lr in CW optimization')
    parser.add_argument('--binary_step', type=int, default=10, metavar='N',
                        help='Binary search step')
    parser.add_argument('--num_iter', type=int, default=100, metavar='N',
                        help='Number of iterations in each search step')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    # parser.add_argument('--feature_transform', action='store_true',
    #                    help="use feature transform")
    args = parser.parse_args()

    if args.dataset == 'Bosphorus':
        num_of_class = 105 + 1
    elif args.dataset == 'Eurecom':
        num_of_class = 52

    if args.model == 'PointNet':
        model = PointNetCls(k=num_of_class, feature_transform=False)
    elif args.model == 'PointNet2_MSG':
        model = PointNet_Msg(num_of_class, normal_channel=False)
    elif args.model == 'PointNet2_SSG':
        model = PointNet_Ssg(num_of_class)
    elif args.model == 'DGCNN':
        model = DGCNN(args,output_channels=num_of_class).to(device)
    else:
        exit('wrong model type')


    model.load_state_dict(
        torch.load(os.path.expanduser(os.path.expanduser(
            '~//yq_pointnet//cls//Bosphorus//DGCNN_model_on_Bosphorus.pth'))))
    test_dataset_path = os.path.expanduser("~//yq_pointnet//BosphorusDB//eval.csv")
    test_set = Bosphorus_Dataset(test_dataset_path)
    model.to(device)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=True,
        num_workers=0)
    # setup attack settings

    if args.adv_func == 'logits':
        adv_func = LogitsAdvLoss(kappa=args.kappa)
    else:
        adv_func = CrossEntropyAdvLoss()
    dist_func = L2Dist()

    # hyper-parameters from their official tensorflow code
    attacker = CW(model, adv_func, dist_func,
                  attack_lr=args.attack_lr,
                  init_weight=10., max_weight=80.,
                  binary_step=args.binary_step,
                  num_iter=args.num_iter)

    '''
    # attack
    test_set = ModelNet40Attack(args.data_root, num_points=args.num_points,
                                normalize=True)
    test_sampler = DistributedSampler(test_set, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=4,
                             pin_memory=True, drop_last=False,
                             sampler=test_sampler)
    '''

    # run attack
    attacked_data, real_label, target_label, success_num = attack(attacker)

    # accumulate results
    data_num = len(test_set)
    success_rate = float(success_num) / float(data_num)
    print("data num: ", data_num)
    print("success rate: ", success_rate)

