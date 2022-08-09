
"""Targeted point perturbation attack."""

import os
import random

from tqdm import tqdm
import argparse
import numpy as np
import pandas
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

import sys
random.seed(7122)
sys.path.append('/home/yjli/yq_pointnet/')
from dataset.bosphorus_dataset import Bosphorus_Dataset
# from config import BEST_WEIGHTS
# from config import MAX_PERTURB_BATCH as BATCH_SIZE
# from dataset import ModelNet40Attack

from attack.CW.CW_utils.basic_util import str2bool, set_seed
from attack.CW.CW_attack import CW
from attack.CW.CW_utils.adv_utils import CrossEntropyAdvLoss, LogitsAdvLoss
from attack.CW.CW_utils.dist_utils import L2Dist
from attack.CW.CW_utils.dist_utils import ChamferDist, ChamferkNNDist
from model.pointnet import PointNetCls, feature_transform_regularizer
from model.pointnet2_MSG import PointNet_Msg
from model.pointnet2_SSG import PointNet_Ssg
from model.dgcnn import DGCNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def check_num_pc_changed(adv, ori):
    logits_mtx = np.logical_and.reduce(adv == ori, axis=1)
    return np.sum(logits_mtx == False)

def rand_row(array, dim_needed):
    array = np.append(array ,array ,axis=0)
    row_total = array.shape[0]
    row_sequence = np.arange(row_total)
    np.random.shuffle(row_sequence)
    return array[row_sequence[0:dim_needed], :]

def targetattack(attacker, model, args):

    point_cloud_txt_file_name = 'yanjieli.txt'
    input_data_path = os.path.expanduser("~//yq_pointnet//test_face_data/" + point_cloud_txt_file_name)
    out_folder = f"/home/yjli/yq_pointnet/attack/CW/AdvData/{args.model}/target"
    if args.whether_renormalization == True:
        out_folder = out_folder + "_renormalization"

    if args.whether_1d == True:
        out_folder = out_folder + "_z"
    if args.whether_3Dtransform:
        out_folder = out_folder + "_3Dtransform"
    if args.whether_resample:
        out_folder = out_folder + "_resample"
    out_folder = out_folder + "_" + args.dist_function
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    ipt_ori = np.loadtxt(input_data_path ,delimiter=' ')
    print(np.shape(ipt_ori))
    # ipt, mu, st = normalization(ipt)
    ipt_ori = rand_row(ipt_ori, 4000)
    ipt = ipt_ori[:, 0:3]
    ipt_c4c5 = ipt_ori[:, 3:5]
    # ipt = np.append(ipt, ipt, axis=0)
    mean = np.expand_dims(np.mean(ipt, axis=0), 0)
    ipt = ipt - mean
    var = np.max(np.sqrt(np.sum(ipt ** 2, axis=1)), 0)
    ipt = ipt / var  # scale
    ipt = np.expand_dims(ipt, 0)
    ipt = torch.from_numpy(ipt).float()
    # ipt = ipt.permute(0, 2, 1)
    ipt = ipt.to(device)
    ipt.requires_grad_(True)
    pc, originlabel = ipt, torch.tensor([105], dtype=torch.float).to(device)
    succ_num = 0
    count =0
    sum1 = 0
    sum2 = 0
    for target in range(0, 10):
        count = count + 1
        target_label = torch.tensor([target])
        pc, target_label = pc.to(device=device, dtype=torch.float), target_label.cuda().float()
        # attack!
        _, best_pc, success_num = attacker.attack(pc, target_label)

        if success_num >= 0:
            adv = torch.Tensor(best_pc).cuda()
            logits, _, _ = model(torch.transpose(adv, 1, 2))
            # print
            pred = torch.argmax(logits, dim=1)  # [B]
            print("adv pred:", pred)
            if pred == target:
                sum1 = sum1 + 1
            best_pc = best_pc.squeeze(0)  # [1,4000, 3] -> [4000,3]
            best_pc = best_pc * var + mean  # denormalization
            result = best_pc
            result = np.concatenate((result, ipt_c4c5), axis=1)

            adv_f = 'adv_' + point_cloud_txt_file_name[:-4] + '_' + str(target) + '.txt'
            adv_fname = os.path.join(out_folder, adv_f)

            np.savetxt(adv_fname, result, fmt='%.04f')
            print("adv data saved in {}".format(adv_fname))

            f = open(adv_fname)
            lines = f.readlines()
            A_row = 0
            A = np.zeros((4000, 3), dtype=float)
            for line in lines:
                list = line.strip('\n').split(' ')
                A[A_row:] = list[0:3]
                A_row += 1
            mean = np.expand_dims(np.mean(A, axis=0), 0)
            A = A - mean
            var = np.max(np.sqrt(np.sum(A ** 2, axis=1)), 0)
            A = A / var
            # read_adv = pandas.read_csv(adv_fname, delimiter=' ')
            A = torch.Tensor(A).unsqueeze(0).cuda().transpose(1, 2)
            logits, _, _ = model(A)
            # # print
            pred = torch.argmax(logits, dim=1)  # [B]
            print("adv pred after renormalization:", pred)
            if pred == target:
                sum2 = sum2 + 1
            # """
            # os.system('conda activate python35')
            # os.system("python structureattack.py")
            # os.system('conda activate cuda11')
            # """
            succ_num += 1
    print("succ sum,", succ_num)
    print(" attack success rate after reinput:", sum1)
    print("  attack success rate after renormalization:", sum2)
    return succ_num, count


def untargetattack(attacker):
    out_folder = "/home/yjli/yq_pointnet/attack/CW/AdvData/PointnetMSG/untarget/"
    test_dataset_path = os.path.expanduser("~//yq_pointnet//BosphorusDB//test.csv")
    test_dataset = Bosphorus_Dataset(test_dataset_path)
    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4)
    num = 0
    for i, data in enumerate(testdataloader, 0):
        points, label = data
        points, label = points.to(device='cuda', dtype=torch.float), label.to(device='cuda', dtype=torch.long)
        # attack!
        _, resultpc, success_num = attacker.attack(points, origin_label=label[0])
        if success_num > 0:
            label = label[0][0].cpu().numpy()
            adv_f = 'untarget_' + "orilabel" + '_' + str(label) + '.txt'
            adv_fname = os.path.join(out_folder, adv_f)
            np.savetxt(adv_fname, resultpc[0], fmt='%.04f')
            print("adv data saved in {}".format(adv_fname))

            """
            os.system('conda activate python35')
            os.system("python structureattack.py")
            os.system('conda activate cuda11')
            """
            num += 1
    return num


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--model', type=str, default='PointNet', metavar='N',
                        choices=['PointNet', 'PointNet++Msg', 'PointNet++Ssg',
                                 'DGCNN'],
                        help='Model to use, [pointnet, pointnet++msg,pointnet++ssg, dgcnn]')
    parser.add_argument('--feature_transform', type=str2bool, default=False,
                        help='whether to use STN on features in PointNet')
    parser.add_argument(
        '--dataset', type=str, default='Bosphorus', help="dataset: Bosphorus | Eurecom")
    parser.add_argument('--dropout', type=float, default=0.5, help='parameters in DGCNN: dropout rate')
    parser.add_argument('--batch_size', type=int, default=-1, metavar='BS',
                        help='Size of batch')
    parser.add_argument('--num_points', type=int, default=4000,
                        help='num of points to use')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings in DGCNN')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use in DGCNN')
    parser.add_argument('--adv_func', type=str, default='logits',
                        choices=['logits', 'cross_entropy'],
                        help='Adversarial loss function to use')
    parser.add_argument('--kappa', type=float, default=30.,
                        help='min margin in logits adv loss')
    parser.add_argument('--attack_lr', type=float, default=0.01,
                        help='lr in CW optimization')
    parser.add_argument('--binary_step', type=int, default=3, metavar='N',
                        help='Binary search step')
    parser.add_argument('--num_iter', type=int, default=200, metavar='N',
                        help='Number of iterations in each search step')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--whether_1d', default=True, type=bool,
                        help='True for z perturbation, False for xyz perturbation')
    parser.add_argument('--whether_target', default=True, type=bool,
                        help='True for target attack, False for untarget attack')
    parser.add_argument('--whether_renormalization', default=True, type=bool,
                        help='True for target attack, False for untarget attack')
    parser.add_argument('--whether_3Dtransform', default=True, type=bool,
                        help='3D transformation')
    parser.add_argument('--dist_function', default="Chamfer", type=str,
                        help='L2dist, Chamfer, ChamferkNN')
    parser.add_argument('--whether_resample', default=True, type=bool,
                        help='L2dist, Chamfer, ChamferkNN')

    args = parser.parse_args()

    if args.dataset == 'Bosphorus' and args.model == 'PointNet':
        num_of_class = 105 + 2
    elif args.dataset == 'Bosphorus':
        num_of_class = 105 + 1
    elif args.dataset == 'Eurecom':
        num_of_class = 52

    if args.model == 'PointNet':
        model = PointNetCls(k=num_of_class, feature_transform=False)
    elif args.model == 'PointNet++Msg':
        model = PointNet_Msg(num_of_class, normal_channel=False)
    elif args.model == 'PointNet++Ssg':
        model = PointNet_Ssg(num_of_class)
    elif args.model == 'DGCNN':
        model = DGCNN(args, output_channels=num_of_class).to(device)
    else:
        exit('wrong model type')

    model.load_state_dict(
        torch.load(os.path.expanduser(os.path.expanduser(
            '~//yq_pointnet//cls//Bosphorus//{}_model_on_Bosphorus.pth'.format(args.model)))))
    # test_dataset_path = os.path.expanduser("~//yq_pointnet//BosphorusDB//eval.csv")
    # test_set = Bosphorus_Dataset(test_dataset_path)
    model.to(device)
    """
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=True,g
        num_workers=0)
    """

    # setup attack settings

    if args.adv_func == 'logits':
        adv_func = LogitsAdvLoss(kappa=args.kappa)
    else:
        adv_func = CrossEntropyAdvLoss()

    if args.dist_function == 'L2dist':
        dist_func = L2Dist()
    elif args.dist_function == 'Chamfer':
        dist_func = ChamferDist()
    elif args.dist_function == 'ChamferkNN':
        dist_func = ChamferkNNDist()
    # hyper-parameters from their official tensorflow code
    attacker = CW(model, adv_func, dist_func,
                  attack_lr=args.attack_lr,
                  init_weight=10., max_weight=80.,
                  binary_step=args.binary_step,
                  num_iter=args.num_iter,
                  whether_target=args.whether_target,
                  whether_renormalization=args.whether_renormalization,
                  whether_3Dtransform=args.whether_3Dtransform,
                  whether_resample=args.whether_resample)

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
    if args.whether_target:
        success_num, count = targetattack(attacker, model, args)
    else:
        success_num, count = untargetattack(attacker)

    # accumulate results
    # data_num = len(test_set)
    data_num = 105
    success_rate = float(success_num) / float(count)
    # print("data num: ", data_num)
    print("success rate: ", success_rate)

