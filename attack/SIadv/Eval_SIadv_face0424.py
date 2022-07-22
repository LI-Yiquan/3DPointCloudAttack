# -*- coding: utf-8 -*-
import os
import argparse
import time
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset


#from utils.logging import Logging_str
from attack.SIadv.utils.utils import set_seed

from SIadv_attack import PointCloudAttack
from attack.SIadv.utils.set_distance import ChamferDistance, HausdorffDistance
from dataset.bosphorus_dataset import Bosphorus_Dataset
from dataset.eurecom_dataset import Eurecom_Dataset

def data_preprocess(data):
    """Preprocess the given data and label.
    """
    points, target = data

    points = points # [B, N, C]

    target = target[:, 0] # [B]

    points = points.cuda()
    target = target.cuda()

    return points, target


def save_tensor_as_txt(points, filename):
    """Save the torch tensor into a txt file.
    """
    points = points.squeeze(0).detach().cpu().numpy()
    with open(filename, "a") as file_object:
        for i in range(points.shape[0]):
            # msg = str(points[i][0]) + ' ' + str(points[i][1]) + ' ' + str(points[i][2])
            msg = str(points[i][0]) + ' ' + str(points[i][1]) + ' ' + str(points[i][2]) + \
                ' ' + str(points[i][3].item()) +' ' + str(points[i][3].item()) + ' '+ str(1-points[i][3].item())
            file_object.write(msg+'\n')
        file_object.close()
    print('Have saved the tensor into {}'.format(filename))


def main():
    # load data
    #test_loader = load_data(args)
    if args.dataset == 'Bosphorus':
        dataset_path = os.path.expanduser("~//yq_pointnet//BosphorusDB//train.csv")
        test_dataset_path = os.path.expanduser("~//yq_pointnet//BosphorusDB//eval.csv")
        dataset = Bosphorus_Dataset(dataset_path)
        print('Bosphorus')
        test_dataset = Bosphorus_Dataset(test_dataset_path)
        num_of_class = 105 + 1


    elif args.dataset == 'Eurecom':
        dataset_path = os.path.expanduser("~//yq_pointnet//EURECOM_Kinect_Face_Dataset//train.csv")
        test_dataset_path = os.path.expanduser("~//yq_pointnet//EURECOM_Kinect_Face_Dataset//eval.csv")
        dataset = Eurecom_Dataset(dataset_path)
        print('Eurecom')
        test_dataset = Eurecom_Dataset(test_dataset_path)
        num_of_class = 52

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.num_workers))

    num_class = 0
    if args.dataset == 'Bosphorus':
        num_class = 105+1
    elif args.dataset == 'Eurecom':
        num_class = 52
    assert num_class != 0
    args.num_class = num_class

    # load model
    attack = PointCloudAttack(args)

    # start attack
    atk_success = 0
    avg_query_costs = 0.
    avg_mse_dist = 0.
    avg_chamfer_dist = 0.
    avg_hausdorff_dist = 0.
    avg_time_cost = 0.
    chamfer_loss = ChamferDistance()
    hausdorff_loss = HausdorffDistance()
    for batch_id, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        # prepare data for testing
        points, target = data
        target = target[:, 0]
        print('\noriginal label: ', target)
        '''
        print("\norignal label: ", label)
        points = points.cuda()
        target = torch.FloatTensor([random.randint(0, num_class-1) for p in range(args.batch_size)])
        while label[:,0]==target:
            target = torch.FloatTensor([random.randint(0, num_class-1) for p in range(args.batch_size)])
        '''
        points = points.cuda()
        target = target.cuda()

        points = points.float()
        # target = target.long()

        # start attack
        t0 = time.clock()
        adv_points, adv_target, query_costs = attack.run(points, target)
        t1 = time.clock()
        avg_time_cost += t1 - t0
        if not args.query_attack_method is None:
            print('>>>>>>>>>>>>>>>>>>>>>>>')
            print('Query cost: ', query_costs)
            print('>>>>>>>>>>>>>>>>>>>>>>>')
            avg_query_costs += query_costs

        atk_success += 1 if adv_target != target else 0
        print('success: ', atk_success)

        # modified point num count
        points = points[:,:,:3].data # P, [1, N, 3]
        pert_pos = torch.where(abs(adv_points-points).sum(2))
        count_map = torch.zeros_like(points.sum(2))
        count_map[pert_pos] = 1.
        # print('Perturbed point num:', torch.sum(count_map).item())

        avg_mse_dist += np.sqrt(F.mse_loss(adv_points, points).detach().cpu().numpy() * 3072)
        avg_chamfer_dist += chamfer_loss(adv_points, points)
        avg_hausdorff_dist += hausdorff_loss(adv_points, points)

    atk_success /= batch_id + 1
    print('Attack success rate: ', atk_success)
    avg_time_cost /= batch_id + 1
    print('Average time cost: ', avg_time_cost)
    if not args.query_attack_method is None:
        avg_query_costs /= batch_id + 1
        print('Average query cost: ', avg_query_costs)
    avg_mse_dist /= batch_id + 1
    print('Average MSE Dist:', avg_mse_dist)
    avg_chamfer_dist /= batch_id + 1
    print('Average Chamfer Dist:', avg_chamfer_dist.item())
    avg_hausdorff_dist /= batch_id + 1
    print('Average Hausdorff Dist:', avg_hausdorff_dist.item())





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Shape-invariant 3D Adversarial Point Clouds')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--input_point_nums', type=int, default=1024,
                        help='Point nums of each point cloud')
    parser.add_argument('--seed', type=int, default=2022, metavar='S',
                        help='random seed (default: 2022)')
    parser.add_argument('--dataset', type=str, default='Bosphorus',
                        choices=['Bosphorus', 'Eurecom'])
    parser.add_argument('--normal', action='store_true', default=False,
                        help='Whether to use normal information [default: False]')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Worker nums of data loading.')
    parser.add_argument('--transfer_attack_method', type=str, default=None,
                        choices=['ifgm_ours'])
    parser.add_argument('--query_attack_method', type=str, default='ours',
                        choices=['simbapp', 'simba', 'ours'])
    parser.add_argument('--surrogate_model', type=str, default='PointNet',
                        choices=['PointNet','PointNet++Msg','DGCNN','CurveNet'])
    parser.add_argument('--target_model', type=str, default='DGCNN',
                        choices=['PointNet','PointNet++Msg','DGCNN','CurveNet'])
    parser.add_argument('--defense_method', type=str, default=None,
                        choices=['sor', 'srs', 'dupnet'])
    parser.add_argument('--top5_attack', action='store_true', default=False,
                        help='Whether to attack the top-5 prediction [default: False]')
    parser.add_argument('--max_steps', default=50, type=int,
                        help='max iterations for black-box attack')
    parser.add_argument('--eps', default=0.16, type=float,
                        help='epsilon of perturbation')
    parser.add_argument('--step_size', default=0.07, type=float,
                        help='step-size of perturbation')
    parser.add_argument(
        '--dropout', type=float, default=0.5, help='parameters in DGCNN: dropout rate')
    parser.add_argument(
        '--k', type=int, default=20, help='parameters in DGCNN: k')
    parser.add_argument(
        '--emb_dims', type=int, default=1024, metavar='N', help='parameters in DGCNN: Dimension of embeddings')
    args = parser.parse_args()

    # basic configuration
    set_seed(args.seed)
    args.device = torch.device("cuda")

    # main loop
    main()