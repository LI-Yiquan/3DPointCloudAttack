import argparse

import CTA
import torch
import os
import numpy as np
from attack.CTA.utils import dis_utils_numpy as dun
from torch import nn
from model.pointnet import PointNetCls, feature_transform_regularizer
from model.pointnet2_MSG import PointNet_Msg
from model.pointnet2_SSG import PointNet_Ssg
from model.dgcnn import DGCNN
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

'''
change the txt file name to generate adv_data
(need to put the txt file in directory : '~/yq_pointnet/test_face_data/')
(the adv_data will also be put in directory: '~/yq_pointnet/test_face_data/')
'''
point_cloud_txt_file_name = 'face0424smile1.txt'


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

def normalization(ipt):
    mu = np.expand_dims(np.mean(ipt, axis=0), 0)
    ipt = ipt - np.expand_dims(np.mean(ipt, axis=0), 0)  # center
    dist = np.max(np.sqrt(np.sum(ipt ** 2, axis=1)), 0)
    std = dist
    point_cloud_data = ipt / dist  # scale

    return point_cloud_data, mu, std

def rand_row(array, dim_needed):
    row_total = array.shape[0]
    row_sequence = np.arange(row_total)
    np.random.shuffle(row_sequence)
    return array[row_sequence[0:dim_needed], :]


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=10, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument(
    '--nepoch', type=int, default=150, help='number of epochs to train for')
parser.add_argument(
    '--outf', type=str, default='cls', help='output folder')
parser.add_argument(
    '--dropout', type=float, default=0.5, help='parameters in DGCNN: dropout rate')
parser.add_argument(
    '--k', type=int, default=20, help='parameters in DGCNN: k')
parser.add_argument(
    '--emb_dims', type=int, default=1024, metavar='N', help='parameters in DGCNN: Dimension of embeddings')
parser.add_argument(
    '--model', type=str, default='DGCNN', help='model type: PointNet or PointNet++ or DGCNN')
parser.add_argument(
    '--dataset', type=str, default='Bosphorus', help="dataset: Bosphorus | Eurecom")
parser.add_argument(
    '--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()

if opt.dataset == 'Bosphorus':
    num_of_class = 105+1
elif opt.dataset == 'Eurecom':
    num_of_class = 52


if opt.model == 'PointNet':
    classifier = PointNetCls(k=num_of_class, feature_transform=False)
elif opt.model == 'PointNet++Msg':
    classifier = PointNet_Msg(num_of_class, normal_channel=False)
elif opt.model == 'PointNet++Ssg':
    classifier = PointNet_Ssg(num_of_class)
elif opt.model == 'DGCNN':
    classifier = DGCNN(opt, output_channels=num_of_class).to(device)


classifier.load_state_dict(torch.load(os.path.expanduser(os.path.expanduser('~//yq_pointnet//cls//Bosphorus//DGCNN_model_on_Bosphorus.pth'))))
layer_name = list(classifier.state_dict().keys())
# selected_layer = 'fc3'
selected_layer = 'linear3'
classifier.to(device)

test_data_path = os.path.expanduser("~//yq_pointnet//test_face_data/"+point_cloud_txt_file_name)


ipt = np.loadtxt(test_data_path, delimiter=',')
ipt = ipt[:,0:3]
ipt = rand_row(ipt, 4000)
ipt, mu, std = normalization(ipt)
ipt = np.expand_dims(ipt, 0)
ipt = np.append(ipt, ipt, axis=0)

ipt = torch.from_numpy(ipt).float()
ipt = ipt.permute(0, 2, 1)
ipt = ipt.to(device)
ipt.requires_grad_(True)

activation_dictionary = {}
# classifier.fc3.register_forward_hook(CTA.layer_hook(activation_dictionary, selected_layer))
classifier.linear3.register_forward_hook(CTA.layer_hook(activation_dictionary, selected_layer))
IG_steps = 25
alpha = torch.tensor(1e-6)
delta = torch.tensor(1)
n_points = 0
verbose = True  # print activation every step
using_softmax_neuron = False  # whether to optimize the unit after softmax
penalize_dis = True
sec_act_noise = False
beta = torch.tensor(1e-5)  # Weighting for penalizing distance
optimizer = 'Adam'
state, output, ori_logits, max_other_logits = CTA.act_max(network=classifier,
                                                          input=ipt,
                                                          layer_activation=activation_dictionary,
                                                          layer_name=selected_layer,
                                                          ori_cls=105,
                                                          IG_steps=IG_steps,
                                                          alpha=alpha,
                                                          beta=beta,
                                                          n_points=n_points,
                                                          verbose=verbose,
                                                          using_softmax_neuron=using_softmax_neuron,
                                                          penalize_dis=penalize_dis,
                                                          optimizer=optimizer,
                                                          target_att=False
                                                          )
res = output.permute(0, 2, 1)
ipt = ipt.permute(0, 2, 1)
if torch.cuda.is_available() == True:
    res = res.cpu().detach().numpy()
else:
    res = res.detach().numpy()
res = res[0]
if torch.cuda.is_available() == True:
    ipt = ipt.cpu().detach().numpy()
else:
    ipt = ipt.detach().numpy()
ipt = ipt[0]

res = res*std + mu
print(res)
if state == 'Suc':
    data_root = os.path.expanduser("~//yq_pointnet//test_face_data/")
    adv_f = 'adv_'+ point_cloud_txt_file_name
    adv_fname = os.path.join(data_root, adv_f)
    np.savetxt(adv_fname, res, fmt='%.04f')
    normalized_f = 'normalized_'+ point_cloud_txt_file_name
    normalized_fname = os.path.join(data_root, normalized_f)
    np.savetxt(normalized_fname, ipt, fmt='%.04f')
    print("Generate point cloud successful!")

    res = np.float32(res)
    target_pro = np.float32(np.squeeze(ipt))
    #Hausdorff_dis2 = dun.bid_hausdorff_dis(res, target_pro)
    #cham_dis = dun.chamfer(res, target_pro)
    num_preturbed_pc = check_num_pc_changed(res, target_pro)

    print('Finding one-point advserial example Successful!')
    #print('Hausdorff distance: ', "%e" % Hausdorff_dis2)
    #print('Chamfer distance: ', "%e" % cham_dis)
    print('Number of points changed: ', num_preturbed_pc)

else:
    print('Finding one-point advserial example failed!')