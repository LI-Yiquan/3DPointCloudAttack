import os
import random
from tqdm import tqdm
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset.bosphorus_dataset import Bosphorus_Dataset
from attack.CW.CW_utils.basic_util import str2bool, set_seed
from attack.KNN.KNN_attack import CWKNN
from attack.CW.CW_utils.adv_utils import CrossEntropyAdvLoss, LogitsAdvLoss, UntargetedLogitsAdvLoss
from attack.CW.CW_utils.dist_utils import L2Dist, ClipPointsLinf
from model.curvenet import CurveNet
from model.pointnet import PointNetCls, feature_transform_regularizer
from model.pointnet2_MSG import PointNet_Msg
from model.pointnet2_SSG import PointNet_Ssg
from model.dgcnn import DGCNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def attack():
    model.eval()
    all_adv_pc = []
    all_real_lbl = []
    all_target_lbl = []
    num = 0
    trans_num = 0
    for i, data in tqdm(enumerate(test_loader, 0)):
        pc, label = data
        target = torch.tensor([label])
        pc, target_label = pc.to(device='cuda', dtype=torch.float), target.cuda().float()

        # attack!
        best_pc, success_num = attacker.attack(pc, target_label)

        #data_root = os.path.expanduser("~//yq_pointnet//attack/CW/AdvData/PointNet")
        #adv_f = '{}-{}-{}.txt'.format(i, int(label.detach().cpu().numpy()), int(target_label.detach().cpu().numpy()))
        #adv_fname = os.path.join(data_root, adv_f)
        #if success_num == 1:
        #    np.savetxt(adv_fname, best_pc.squeeze(0), fmt='%.04f')
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
    parser.add_argument('--attack_method', type=str, default='untarget', help="untarget | top1_error")
    parser.add_argument('--model', type=str, default='PointNet', metavar='N',
                        help="Model to use, ['PointNet', 'PointNet++Msg','DGCNN', 'CurveNet']")
    parser.add_argument('--trans_model', type=str, default='PointNet++Msg', metavar='N',
                        help="Model to use, ['PointNet', 'PointNet++Msg','DGCNN', 'CurveNet']")
    parser.add_argument('--dataset', type=str, default='Bosphorus',
                        help='dataset : Bosphorus | Eurecom')
    parser.add_argument('--feature_transform', type=str2bool, default=False,
                        help='whether to use STN on features in PointNet')
    parser.add_argument('--dropout', type=float, default=0.5, help='parameters in DGCNN: dropout rate')
    parser.add_argument('--batch_size', type=int, default=1, metavar='BS',
                        help='Size of batch')
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
    parser.add_argument('--attack_lr', type=float, default=1e-2,
                        help='lr in CW optimization')
    parser.add_argument('--binary_step', type=int, default=1, metavar='N',
                        help='Binary search step')
    parser.add_argument('--num_iter', type=int, default=100, metavar='N',
                        help='Number of iterations in each search step')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--num_of_class', default=105+1, type=int,
                        help='number of class')
    parser.add_argument('--budget', default=0.18,type=float,
                        help='budget parameter in the clip function, use 0.18 | 0.45')
    # parser.add_argument('--feature_transform', action='store_true',
    #                    help="use feature transform")
    args = parser.parse_args()


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

    model.load_state_dict(
        torch.load(os.path.expanduser(
            '~//yq_pointnet//cls//{}//{}_model_on_{}.pth'.format(args.dataset, args.model, args.dataset))))
    model.eval()
    model.to(device)

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

    trans_model.load_state_dict(
        torch.load(os.path.expanduser(
            '~//yq_pointnet//cls//{}//{}_model_on_{}.pth'.format(args.dataset, args.trans_model, args.dataset))))
    trans_model.eval()
    trans_model.to(device)

    test_dataset_path = os.path.expanduser("~//yq_pointnet//BosphorusDB//eval.csv")
    test_set = Bosphorus_Dataset(test_dataset_path)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=True,
        num_workers=0)


    if args.adv_func == 'logits':
        adv_func = LogitsAdvLoss(kappa=args.kappa)
    else:
        adv_func = CrossEntropyAdvLoss()

    if args.attack_method == 'untarget':
        adv_func = UntargetedLogitsAdvLoss(kappa=args.kappa)


    dist_func = L2Dist()
    clip_func = ClipPointsLinf(budget=args.budget)


    # hyper-parameters from their official tensorflow code
    attacker = CWKNN(model=model, trans_model=trans_model,adv_func=adv_func, dist_func=dist_func,
                  attack_lr=args.attack_lr,
                  num_iter=args.num_iter,clip_func=clip_func,attack_method=args.attack_method)

    # run attack
    attacked_data, real_label, target_label, success_num= attack()

    # accumulate results
    data_num = len(test_set)
    success_rate = float(success_num) / float(data_num)
    print("data num: ", data_num)



