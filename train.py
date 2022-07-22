from __future__ import print_function
import argparse
import os
import random
import sys
import torch.optim as optim
import torch.utils.data
from model.pointnet import PointNetCls, feature_transform_regularizer
from model.pointnet2_MSG import PointNet_Msg
from model.pointnet2_SSG import PointNet_Ssg
from model.dgcnn import DGCNN
import torch.nn.functional as F
from dataset.bosphorus_dataset import Bosphorus_Dataset
from dataset.eurecom_dataset import Eurecom_Dataset
from tqdm import tqdm
from model.curvenet import CurveNet


def cal_loss(pred, gold, smoothing=True):
    """ Calculate cross entropy loss, apply label smoothing if needed. """

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


sys.path.append("../")
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
    '--model', type=str, default='PointNet', help='model type: PointNet or PointNet++Ssg or PointNet++Msg or '
                                                       'DGCNN or CurveNet')
parser.add_argument(
    '--dataset', type=str, default='Bosphorus', help="dataset: Bosphorus | Eurecom")
parser.add_argument(
    '--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# print(device)

opt.manualSeed = random.randint(1, 10000)  # fix seed
# opt.manualSeed = 7122
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
num_of_class = 105
if opt.dataset == 'Bosphorus':
    dataset_path = os.path.expanduser("~//yq_pointnet//BosphorusDB//train.csv")
    test_dataset_path = os.path.expanduser("~//yq_pointnet//BosphorusDB//eval.csv")
    dataset = Bosphorus_Dataset(dataset_path)
    print('Bosphorus')
    test_dataset = Bosphorus_Dataset(test_dataset_path)
    num_of_class = 105 + 1


elif opt.dataset == 'Eurecom':
    dataset_path = os.path.expanduser("~//yq_pointnet//EURECOM_Kinect_Face_Dataset//train.csv")
    test_dataset_path = os.path.expanduser("~//yq_pointnet//EURECOM_Kinect_Face_Dataset//eval.csv")
    dataset = Eurecom_Dataset(dataset_path)
    print('Eurecom')
    test_dataset = Eurecom_Dataset(test_dataset_path)
    num_of_class = 52

else:
    exit('wrong dataset')

if opt.model == 'PointNet':
    classifier = PointNetCls(k=num_of_class, feature_transform=opt.feature_transform)

elif opt.model == 'PointNet++Msg':
    classifier = PointNet_Msg(num_of_class, normal_channel=False)

elif opt.model == 'PointNet++Ssg':
    classifier = PointNet_Ssg(num_of_class)

elif opt.model == 'DGCNN':
    classifier = DGCNN(opt, output_channels=num_of_class).to(device)

elif opt.model == 'CurveNet':
    classifier = CurveNet(num_classes=num_of_class)

else:
    exit('wrong model type')

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
# num_classes = len(dataset.classes)
# print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier.to(device)
# classifier.load_state_dict(
#        torch.load(os.path.expanduser(os.path.expanduser(
#            '~//yq_pointnet//cls//Bosphorus//DGCNN_model_on_Bosphorus.pth'))))
optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.to(device)

num_batch = len(dataset) / opt.batchSize

best = 0
for epoch in range(opt.nepoch):
    scheduler.step()
    count = 0
    count_right = 0
    for i, data in enumerate(dataloader, 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.to(device=device, dtype=torch.float), target.to(device)
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)
        if opt.model == 'CurveNet':
            loss = cal_loss(pred, target)
        else:
            loss = F.nll_loss(pred, target)

        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        if 105 in target:
            print("target:",target)
            print("pred:",pred_choice)
        correct = pred_choice.eq(target.data).cpu().sum()
        count_right = count_right + correct.item()
        count = count + len(target)

    total_correct = 0
    total_testset = 0
    for i, data in tqdm(enumerate(testdataloader, 0)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.to(device=device, dtype=torch.float), target.to(device)
        classifier = classifier.eval()
        pred, _, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        total_correct += correct.item()
        total_testset += points.size()[0]
    print('[%d] train loss: %f train accu: %.2f, test accu: %.2f' % (
        epoch, loss.item(), count_right / float(count), total_correct / float(total_testset)))
    if (total_correct / float(total_testset)) > best:
        best = total_correct / float(total_testset)
        root = os.path.expanduser("~//yq_pointnet")
        if not os.path.isdir(os.path.join(root, '%s/%s' % (opt.outf, opt.dataset))):
            os.makedirs(os.path.join(root, '%s/%s' % (opt.outf, opt.dataset)))
        print("best test acc: {:.4} saved!".format(best))
        #torch.save(classifier.state_dict(), '%s/%s/%s_model_on_%s.pth' % (opt.outf, opt.dataset, opt.model,
        #                                                                  opt.dataset))
