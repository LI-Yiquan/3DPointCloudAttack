import os
import open3d as o3d
import numpy
import numpy as np
import pandas as pd
import torch
from numpy import zeros
from torch.utils.data import Dataset, DataLoader
from utils.readbnt import read_bntfile

data_root = os.path.expanduser('~//yq_pointnet//attack//CW//AdvData//PointNet')

np.set_printoptions(suppress=True)
numpy.set_printoptions(suppress=True)
numpy.set_printoptions(threshold=numpy.inf)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=4)
numpy.set_printoptions(precision=4)


def read_PC(idx, path):
    A = zeros((4000, 3), dtype=float)
    files = os.listdir(path)
    ori = tar = idx
    for file in files:
        id = file.split('-')
        if int(id[0]) == idx:
            ori = int(id[1])
            tar = int(id[2].split('.')[0])
            f = open(path + "/" + file)
            lines = f.readlines()
            A_row = 0
            for line in lines:
                list = line.strip('\n').split(' ')
                A[A_row:] = list[0:3]
                A_row += 1
            break
    return A, ori, tar


class AdvData_Dataset(Dataset):
    """
    Args:
        csv_path (string): Path to the csv file with annotations.
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, data_path):
        self.path = data_path
        self.num_of_classes = 104
        self.len = 1341
        """
        csv_path = os.path.expanduser(csv_path)
        assert os.path.exists(csv_path), '%s not found' % csv_path

        # read csv file
        self.df = pd.read_csv(csv_path, header=0, names=['point_cloud_path', 'cls_name'])
        self.df.cls_name, _ = pd.factorize(self.df.cls_name, sort=True)

        self._num_of_classes = len(self.df.cls_name.drop_duplicates())
        self._len_of_dataset = len(self.df.cls_name)
        self._transform = transform
        """
    def get_num_of_classes(self):
        return self.num_of_classes

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        """
        point_cloud_path,  cls_id = self.df.iloc[idx]
        _, _, point_cloud_data = read_bntfile(point_cloud_path)
        #pcd = o3d.io.read_point_cloud(point_cloud_path)
        #pcd = o3d.geometry.PointCloud.uniform_down_sample(pcd, 4)
        #print("pcd；  ", pcd)
        """
        point_cloud_data, ori, tar = read_PC(idx, self.path)

        return point_cloud_data, ori, tar













