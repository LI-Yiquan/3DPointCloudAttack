from __future__ import print_function
import argparse
import os
import random
import sys
sys.path.append("../")
import torch.optim as optim
import torch.utils.data

from model.pointnet import PointNetCls, feature_transform_regularizer
from model.pointnet2_MSG import PointNet_Msg
from model.pointnet2_SSG import PointNet_Ssg
from model.dgcnn import DGCNN
import torch.nn.functional as F
from dataset.bosphorus_dataset import Bosphorus_Dataset
from dataset.AdvData_dataset import AdvData_Dataset
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=1, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=150, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--dropout', type=float, default=0.5, help='parameters in DGCNN: dropout rate')
parser.add_argument('--k', type=int, default=20, help='parameters in DGCNN: k')
parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='parameters in DGCNN: Dimension of embeddings')
parser.add_argument('--load_model_path', type=str, default='~//yq_pointnet//CW_utils//cls//DGCNN_model_50.pth', help='model path')
parser.add_argument('--model', type=str, default='DGCNN', help='model type: PointNet or PointNet++ or DGCNN')
#parser.add_argument('--dataset', type=str, default='../shapenetcore_partanno_segmentation_benchmark_v0', help="dataset path")
parser.add_argument('--dataset_type', type=str, default='bosphorus', help="dataset type shapenet|modelnet40|bosphorus")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset_path = os.path.expanduser("~//yq_pointnet//BosphorusDB//train.csv")
#test_dataset_path = os.path.expanduser("~//yq_pointnet//BosphorusDB//eval.csv")

dataset = Bosphorus_Dataset(dataset_path)
#test_dataset = Bosphorus_Dataset(test_dataset_path)

test_dataset = AdvData_Dataset(os.path.expanduser('~//yq_pointnet//attack//CW//AdvData//MSG'))

if opt.model == 'PointNet':
    classifier = PointNetCls(k=105, feature_transform=opt.feature_transform)

elif opt.model == 'PointNet++Msg':
    classifier = PointNet_Msg(105, normal_channel=False)

elif opt.model == 'PointNet++Ssg':
    classifier = PointNet_Ssg(105)

elif opt.model == 'DGCNN':
    classifier = DGCNN(opt).to(device)

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
#num_classes = len(dataset.classes)
#print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

#classifier = PointNetCls(k=16, feature_transform=opt.feature_transform)

if opt.load_model_path != '':
    classifier.load_state_dict(torch.load(os.path.expanduser(opt.load_model_path)))

classifier.to(device)

#print(classifier)
#for param in classifier.parameters():
    #param.requires_grad = False

#fc3_features = classifier.fc3.in_features
#classifier.fc3 = nn.Linear(fc3_features, 20)


optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.to(device)

#print(classifier)

num_batch = len(dataset) / opt.batchSize

total_correct = 0
total_testset = 0
for i, data in tqdm(enumerate(testdataloader, 0)):
    points, original, target = data
    #target = target[:, 0]
    target = original
    points = points.transpose(2, 1)
    points, target = points.to(device='cuda', dtype=torch.float), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

print('test accu: %.2f' % (total_correct / float(total_testset)))


"""
for epoch in range(opt.nepoch+1):
    scheduler.step()
    count = 0
    count_right = 0
    for i, data in enumerate(dataloader, 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.to(device='cuda', dtype=torch.float), target.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)
        loss = F.nll_loss(pred, target)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        count_right = count_right + correct.item()
        count = count + len(target)

    total_correct = 0
    total_testset = 0
    for i,data in tqdm(enumerate(testdataloader, 0)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.to(device='cuda', dtype=torch.float), target.cuda()
        classifier = classifier.eval()
        pred, _, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        total_correct += correct.item()
        total_testset += points.size()[0]
    print('[%d] train loss: %f train accu: %.2f, test accu: %.2f' % (epoch, loss.item(), count_right / float(count), total_correct / float(total_testset)))

    if epoch % 10 == 0 :
        torch.save(classifier.state_dict(), '%s/%s_model_%d_416.pth' % (opt.outf,opt.model, epoch))
"""






