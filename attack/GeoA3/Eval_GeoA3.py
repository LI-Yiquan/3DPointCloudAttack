import os
import random
from tqdm import tqdm
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset.bosphorus_dataset import Bosphorus_Dataset

from attack.Gen3DAdv.utils.basic_util import str2bool, set_seed
from model.curvenet import CurveNet
from model.pointnet import PointNetCls, feature_transform_regularizer
from model.pointnet2_MSG import PointNet_Msg
from model.pointnet2_SSG import PointNet_Ssg
from model.dgcnn import DGCNN

from attack.GeoA3.GeoA3_attack import geoA3_attack

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

print(device)
def rand_row(array, dim_needed):

    row_total = array.shape[0]
    row_sequence = np.arange(row_total)
    np.random.shuffle(row_sequence)
    return array[row_sequence[0:dim_needed], :]


def attack():
    model.eval()
    all_adv_pc = []
    all_real_lbl = []
    all_target_lbl = []
    num = 0
    trans_num = 0
    if args.attack_method == 'untarget':
        for i, data in tqdm(enumerate(test_loader, 0)):
            pc, label = data
            target = torch.tensor([label])
            pc, target_label = pc.to(device=device, dtype=torch.float), target.to(device).float()

            # attack!
            _, best_pc, success_num = geoA3_attack(net=pt_model,pt_model=pt_model,ptm_model=ptm_model,pts_model=pts_model,dgcnn_model=dgcnn_model,
                                                   cur_model=cur_model,pc=pc,label=target_label,cfg=args,i=i,loader_len=1)

            # data_root = os.path.expanduser("~//yq_pointnet//attack/CW/AdvData/PointNet")
            # adv_f = '{}-{}-{}.txt'.format(i, int(label.detach().cpu().numpy()), int(target_label.detach().cpu().numpy()))
            # adv_fname = os.path.join(data_root, adv_f)
            # if success_num == 1:
            #     np.savetxt(adv_fname, best_pc.squeeze(0), fmt='%.04f')
            # results
            num += success_num

            all_adv_pc.append(best_pc)
            all_real_lbl.append(label.detach().cpu().numpy())
            all_target_lbl.append(target_label.detach().cpu().numpy())
    else:
        data_root = os.path.expanduser('~//yq_pointnet//AddData//face0424.txt')
        point_cloud_data = np.loadtxt(data_root, delimiter=',')
        point_cloud_data = rand_row(point_cloud_data, 4000)
        point_cloud_data = point_cloud_data[:, 0:3]
        center = np.expand_dims(np.mean(point_cloud_data, axis=0), 0)
        point_cloud_data = point_cloud_data - center  # center
        dist = np.max(np.sqrt(np.sum(point_cloud_data ** 2, axis=1)), 0)
        point_cloud_data = point_cloud_data / dist  # scale
        pc = torch.from_numpy(point_cloud_data.astype(np.float))
        pc = pc.unsqueeze(0)
        for j in range(0, 105):
            label = torch.tensor([105])
            alist = []
            target = j
            alist.append(target)
            target = torch.tensor(alist)
            # target = target[:, 0]
            # pc = pc.transpose(2, 1)

            pc, target_label, label = pc.to(device=device,
                                            dtype=torch.float), target.to(device).float(), label.to(device).float()

            # attack!
            _, best_pc, success_num = geoA3_attack(pt_model,pc,target_label,args, i=1,loader_len=1)

            data_root = os.path.expanduser("~//yq_pointnet//attack/Gen3DAdv/AdvData/{}".format(args.model))
            if not os.path.exists(data_root):
                os.mkdir(data_root)
            adv_f = '{}.txt'.format(int(target_label.detach().cpu().numpy()))
            adv_fname = os.path.join(data_root, adv_f)
            best_pc = best_pc.squeeze(0)
            best_pc = best_pc * dist + center
            if success_num == 1:
                np.savetxt(adv_fname, best_pc, fmt='%.04f')

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
    parser.add_argument('--attack_method', type=str, default='untarget', help="untarget | target")
    parser.add_argument('--model', type=str, default='CurveNet', metavar='N',
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
    parser.add_argument('--kappa', type=float, default=30,
                        help='min margin in logits adv loss')
    parser.add_argument('--attack_lr', type=float, default=1e-2,
                        help='lr in CW optimization')
    parser.add_argument('--binary_step', type=int, default=1, metavar='N',
                        help='Binary search step')
    parser.add_argument('--num_iter', type=int, default=100, metavar='N',
                        help='Number of iterations in each search step')
    parser.add_argument('--num_of_class', default=105+1, type=int,
                        help='number of class')
    parser.add_argument('--budget', default=0.18,type=float,
                        help='budget parameter in the clip function, use 0.18 | 0.45')
    # ------------Dataset-----------------------
    parser.add_argument('--npoint', default=1024, type=int, help='')
    parser.add_argument('--classes', default=105+1, type=int, help='')
    # ------------Attack-----------------------
    parser.add_argument('--attack', default=None, type=str, help='GeoA3 | GeoA3_mesh')
    parser.add_argument('--attack_label', default='All', type=str, help='[All; ...; Untarget]')
    parser.add_argument('--binary_max_steps', type=int, default=10, help='')
    parser.add_argument('--initial_const', type=float, default=10, help='')
    parser.add_argument('--iter_max_steps', default=500, type=int, metavar='M', help='max steps')
    parser.add_argument('--optim', default='adam', type=str, help='adam| sgd')
    parser.add_argument('--lr', type=float, default=0.01, help='')
    parser.add_argument('--eval_num', type=int, default=1, help='')
    ## cls loss
    parser.add_argument('--cls_loss_type', default='CE', type=str, help='Margin | CE')
    parser.add_argument('--confidence', type=float, default=0, help='confidence for margin based attack method')
    ## distance loss
    parser.add_argument('--dis_loss_type', default='CD', type=str, help='CD | L2 | None')
    parser.add_argument('--dis_loss_weight', type=float, default=1.0, help='')
    parser.add_argument('--is_cd_single_side', action='store_true', default=False, help='')
    ## hausdorff loss
    parser.add_argument('--hd_loss_weight', type=float, default=0.1, help='')
    ## normal loss
    parser.add_argument('--curv_loss_weight', type=float, default=1.0, help='')
    parser.add_argument('--curv_loss_knn', type=int, default=16, help='')
    ## uniform loss
    parser.add_argument('--uniform_loss_weight', type=float, default=0.0, help='')
    ## KNN smoothing loss
    parser.add_argument('--knn_smoothing_loss_weight', type=float, default=5.0, help='')
    parser.add_argument('--knn_smoothing_k', type=int, default=5, help='')
    parser.add_argument('--knn_threshold_coef', type=float, default=1.10, help='')
    ## Laplacian loss for mesh
    parser.add_argument('--laplacian_loss_weight', type=float, default=0, help='')
    parser.add_argument('--edge_loss_weight', type=float, default=0, help='')
    ## Mesh opt
    parser.add_argument('--is_partial_var', dest='is_partial_var', action='store_true', default=False, help='')
    parser.add_argument('--knn_range', type=int, default=3, help='')
    parser.add_argument('--is_subsample_opt', dest='is_subsample_opt', action='store_true', default=False, help='')
    parser.add_argument('--is_use_lr_scheduler', dest='is_use_lr_scheduler', action='store_true', default=False,
                        help='')
    ## perturbation clip setting
    parser.add_argument('--cc_linf', type=float, default=0.0, help='Coefficient for infinity norm')
    ## Proj offset
    parser.add_argument('--is_real_offset', action='store_true', default=False, help='')
    parser.add_argument('--is_pro_grad', action='store_true', default=False, help='')
    ## Jitter
    parser.add_argument('--is_pre_jitter_input', action='store_true', default=False, help='')
    parser.add_argument('--is_previous_jitter_input', action='store_true', default=False, help='')
    parser.add_argument('--calculate_project_jitter_noise_iter', default=50, type=int, help='')
    parser.add_argument('--jitter_k', type=int, default=16, help='')
    parser.add_argument('--jitter_sigma', type=float, default=0.01, help='')
    parser.add_argument('--jitter_clip', type=float, default=0.05, help='')
    ## PGD-like attack
    parser.add_argument('--step_alpha', type=float, default=5, help='')
    # ------------Recording settings-------
    parser.add_argument('--is_record_converged_steps', action='store_true', default=False, help='')
    parser.add_argument('--is_record_loss', action='store_true', default=False, help='')
    # ------------OS-----------------------
    parser.add_argument('-j', '--num_workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--is_save_normal', action='store_true', default=False, help='')
    parser.add_argument('--is_debug', action='store_true', default=False, help='')
    parser.add_argument('--is_low_memory', action='store_true', default=False, help='')

    args = parser.parse_args()


    if args.model == 'PointNet':
        model = PointNetCls(k=args.num_of_class, feature_transform=args.feature_transform)
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


    pt_model = PointNetCls(k=args.num_of_class, feature_transform=False)
    pt_model.load_state_dict(
        torch.load(os.path.expanduser(
            '~//yq_pointnet//cls//{}//{}_model_on_{}.pth'.format(args.dataset, 'PointNet', args.dataset))))
    pt_model.eval()
    pt_model.to(device)

    ptm_model = PointNet_Msg(args.num_of_class, normal_channel=False)
    ptm_model.load_state_dict(
        torch.load(os.path.expanduser(
            '~//yq_pointnet//cls//{}//{}_model_on_{}.pth'.format(args.dataset, 'PointNet++Msg', args.dataset))))
    ptm_model.eval()
    ptm_model.to(device)

    pts_model = PointNet_Ssg(args.num_of_class)
    pts_model.load_state_dict(
        torch.load(os.path.expanduser(
            '~//yq_pointnet//cls//{}//{}_model_on_{}.pth'.format(args.dataset, 'PointNet++Ssg', args.dataset))))
    pts_model.eval()
    pts_model.to(device)

    dgcnn_model = DGCNN(args, output_channels=args.num_of_class).to(device)
    dgcnn_model.load_state_dict(
        torch.load(os.path.expanduser(
            '~//yq_pointnet//cls//{}//{}_model_on_{}.pth'.format(args.dataset, 'DGCNN', args.dataset))))
    dgcnn_model.eval()
    dgcnn_model.to(device)

    cur_model = CurveNet(num_classes=args.num_of_class)
    cur_model.load_state_dict(
        torch.load(os.path.expanduser(
            '~//yq_pointnet//cls//{}//{}_model_on_{}.pth'.format(args.dataset, 'CurveNet', args.dataset))))
    cur_model.eval()
    cur_model.to(device)



    test_dataset_path = os.path.expanduser("~//yq_pointnet//BosphorusDB//eval.csv")
    test_set = Bosphorus_Dataset(test_dataset_path)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=True,
        num_workers=0)


    # if args.adv_func == 'logits':
    #     adv_func = LogitsAdvLoss(kappa=args.kappa)
    # else:
    #     adv_func = CrossEntropyAdvLoss()

    # if args.attack_method == 'untarget':
    #     adv_func = UntargetedLogitsAdvLoss(kappa=args.kappa)


    # dist_func = L2Dist()
    # clip_func = ClipPointsLinf(budget=args.budget)



    # run attack
    attacked_data, real_label, target_label, success_num= attack()

    # accumulate results
    data_num = len(test_set)
    success_rate = float(success_num) / float(data_num)
    print("data num: ", data_num)



