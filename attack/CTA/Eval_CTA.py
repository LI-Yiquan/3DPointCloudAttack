import CTA
import argparse
import torch
import os
import sys
sys.path.append("../")
import numpy as np
from attack.CTA.utils import dis_utils_numpy as dun
from dataset.bosphorus_dataset import Bosphorus_Dataset
from model.pointnet import PointNetCls, feature_transform_regularizer
from model.pointnet2_MSG import PointNet_Msg
from model.dgcnn import DGCNN
from torch import nn
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#ROOT_DIR = BASE_DIR
#sys.path.append(os.path.join(ROOT_DIR, 'models'))



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def check_num_pc_changed(adv, ori):
    logits_mtx = np.logical_and.reduce(adv == ori, axis=1)
    return np.sum(logits_mtx == False)


def sampling(points, sample_size):
    num_p = points.shape[0]
    index = range(num_p)
    np.random.seed(1)
    sampled_index = np.random.choice(index, size=sample_size)
    sampled = points[sampled_index]
    return sampled


def generate_adv(prototype, ori_class, filename):
    # =============================================================================
    #     np.save('visu/Ori_pt.npy',np.squeeze(prototype))
    #     ori_pt = o3d.geometry.PointCloud()
    #     ori_pt.points = o3d.utility.Vector3dVector(prototype[0,:,0:3])
    #     o3d.io.write_point_cloud('visu/Ori_pt.ply', ori_pt)
    # =============================================================================
    #ipt = torch.from_numpy(prototype).float()
    ipt = prototype.float()
    ipt = ipt.permute(0, 2, 1)
    ipt.requires_grad_(True)
    activation_dictionary = {}
    classifier.fc3.register_forward_hook(CTA.layer_hook(activation_dictionary, selected_layer))
    # steps = 10              # perform 100 iterations                  # flamingo class of Imagenet
    IG_steps = 50
    if torch.cuda.is_available() == True:
        alpha = torch.tensor(1e-6).cuda()
        beta = torch.tensor(3e-6).cuda()
    else:
        alpha = torch.tensor(1e-6)
        beta = torch.tensor(1e-4)
    verbose = True  # print activation every step
    using_softmax_neuron = False  # whether to optimize the unit after softmax
    penalize_dis = False
    optimizer = 'Adam'
    target_att = 'random'
    state, output, ori_logits, max_other_logits = CTA.act_max(network=classifier,
                                                              input=ipt,
                                                              layer_activation=activation_dictionary,
                                                              layer_name=selected_layer,
                                                              ori_cls=ori_class,
                                                              IG_steps=IG_steps,
                                                              alpha=alpha,
                                                              beta=beta,
                                                              n_points=100,
                                                              verbose=verbose,
                                                              using_softmax_neuron=using_softmax_neuron,
                                                              penalize_dis=penalize_dis,
                                                              optimizer=optimizer,
                                                              target_att=target_att
                                                              )
    res = output.permute(0, 2, 1)
    ipt = ipt.permute(0,2,1)
    if torch.cuda.is_available() == True:
        res = res.detach().cpu().numpy()
        ipt = ipt.detach().cpu().numpy()
    else:
        res = res.detach().numpy()
        ipt = ipt.detach().numpy()

    res = res[0]
    res = np.float32(res)
    ipt = ipt[0]
    ipt = np.float32(ipt)
    ##########################################
    # Save npy for trasferability test
    if state == 'Suc':
        #trans_path = 'transferability/pn1_CTA/'
        #np.save(trans_path + filename, res)

        data_root = os.path.expanduser("~//yq_pointnet//CTA_adv_data/")
        fname = os.path.join(data_root, "{}.txt".format(filename))
        save_data = res
        #print(save_data)
        np.savetxt(fname, save_data, fmt='%.04f')
        ori_fname = os.path.join(data_root, "ori_{}.txt".format(filename))
        ori_save_data = ipt
        np.savetxt(ori_fname, ori_save_data, fmt='%.04f')
    ##########################################

    target_pro = np.float32(np.squeeze(prototype.cpu()))
    Hausdorff_dis2 = dun.bid_hausdorff_dis(res, target_pro)
    cham_dis = dun.chamfer(res, target_pro)
    num_perturbed_pc = check_num_pc_changed(res, target_pro)
    return state, Hausdorff_dis2, cham_dis, num_perturbed_pc


def get_pred(points, classifier):
    #points = torch.from_numpy(points)
    points = points.transpose(2, 1)
    points = points.float()
    if torch.cuda.is_available() == True:
        points = points.cuda()
        classifier = classifier.cuda()
    classifier = classifier.eval()
    pred, bf_sftmx, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    return pred_choice, pred_choice






parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=10, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=40, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--dropout', type=float, default=0.5, help='parameters in DGCNN: dropout rate')
parser.add_argument('--k', type=int, default=20, help='parameters in DGCNN: k')
parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='parameters in DGCNN: Dimension of embeddings')
parser.add_argument('--load_model_path', type=str, default='~//yq_pointnet//CW_utils//cls//cls_model_14.pth', help='model path')
parser.add_argument('--model', type=str, default='PointNet++', help='model type: PointNet or PointNet++ or DGCNN')
#parser.add_argument('--dataset', type=str, default='../shapenetcore_partanno_segmentation_benchmark_v0', help="dataset path")
parser.add_argument('--dataset_type', type=str, default='bosphorus', help="dataset type shapenet|modelnet40|bosphorus")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

if opt.model == 'PointNet':
    classifier = PointNetCls(k=105, feature_transform=opt.feature_transform)

elif opt.model == 'PointNet++':
    classifier = PointNet_Msg(105, normal_channel=False)

elif opt.model == 'DGCNN':
    classifier = DGCNN(opt).to(device)

else:
    exit('wrong model type')


classifier = PointNetCls(k=105, feature_transform=True)


classifier.load_state_dict(torch.load(os.path.expanduser(os.path.expanduser('~//yq_pointnet//CW_utils//cls//yq_cls_model_150.pth'))))
test_dataset_path = os.path.expanduser("~//yq_pointnet//BosphorusDB//eval.csv")
test_dataset = Bosphorus_Dataset(test_dataset_path)


classifier.to(device)
testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0)


#if torch.cuda.is_available() == True:
#    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
#else:
#    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', map_location=torch.device('cpu'))
#print(classifier.load_state_dict(checkpoint['model_state_dict']))
layer_name = list(classifier.state_dict().keys())
selected_layer = 'fc3'

num_class = 105
total_num_ins = 0
total_suc_num = 0
total_avg_Hausdorff_dis = 0
total_avg_cham_dis = 0
class_num_ins = np.zeros((num_class))
class_suc_num = np.zeros((num_class))
class_avg_Hausdorff_dis = np.zeros((num_class))
class_avg_cham_dis = np.zeros((num_class))
avg_number_points_changed = 0
for i, data in enumerate(testdataloader, 0):
    points, target = data
    ori_class = target
    #if torch.isnan(points).any:
    #    continue
    points, target = points.to(device), target.to(device)
    #prototype = np.expand_dims((sampling(points, 1024)), 0)
    prototype = points
    cur_cls_num, pred_class = get_pred(prototype, classifier)
    ori_class, pred_class = ori_class.to(device), pred_class.to(device)
    if ori_class == pred_class:  # Only generate instances being classified correctly
        #print(ori_class)
        #print(pred_class)
        class_num_ins[cur_cls_num] += 1
        total_num_ins += 1
        #save_name = ori_class + str(int(class_num_ins[cur_cls_num]))
        save_name = pred_class
        #print(save_name.cpu().numpy()[0])
        #saving_path = 'visu/output/second/'
        saving_path = os.path.expanduser("~//yq_pointnet//CTA_adv_data/")
        if ((str)(save_name.cpu().numpy()[0]) + '.ply') in os.listdir(saving_path):
            print('Already processed, skip!')
            continue


        if torch.cuda.is_available() == True:
            state, Hausdorff_dis, cham_dis, num_perturbed_pc = generate_adv(prototype,
                                                                            cur_cls_num.detach().cpu().numpy()[0],
                                                                            save_name)
        else:
            state, Hausdorff_dis, cham_dis, num_perturbed_pc = generate_adv(prototype, cur_cls_num.detach().numpy()[0],
                                                                            save_name)

        if state == 'Suc':
            class_suc_num[cur_cls_num] += 1
            total_suc_num += 1
            class_avg_Hausdorff_dis[cur_cls_num] += Hausdorff_dis
            class_avg_cham_dis[cur_cls_num] += cham_dis
            total_avg_Hausdorff_dis += Hausdorff_dis
            total_avg_cham_dis += cham_dis
            avg_number_points_changed += num_perturbed_pc
            print('Hausdorff distance: ', "%e" % Hausdorff_dis)
            print('Chamfer distance: ', "%e" % cham_dis)
            print('Number of points changed: ', num_perturbed_pc)
        elif state == 'Fail':
            print('Finding adversarial example failed!')
class_suc_rate = class_suc_num / class_num_ins
class_avg_Hausdorff_dis = class_avg_Hausdorff_dis / class_suc_num
class_avg_cham_dis = class_avg_cham_dis / class_suc_num
print('\n')
print('*****************************************************************')
print('Average Class Hausdorff Distance :', class_avg_Hausdorff_dis)
print('Average Class Chamfer Distance :', class_avg_cham_dis)
print('Class Success percentage:', class_suc_rate)
print('*****************************************************************')
print('\n')
total_avg_Hausdorff_dis /= total_suc_num
total_avg_cham_dis /= total_suc_num
total_suc_rate = total_suc_num / total_num_ins
avg_number_points_changed = avg_number_points_changed / total_suc_num
print('\n')
print('##################################################################')
print('Total number of instance tested: ', total_num_ins)
print('Total Average Hausdorff Distance :', total_avg_Hausdorff_dis)
print('Total Average Chamfer Distance :', total_avg_cham_dis)
print('Total Average number of points perturbed :', avg_number_points_changed)
print('Total Success percentage:', total_suc_rate)
print('##################################################################')
print('\n')
