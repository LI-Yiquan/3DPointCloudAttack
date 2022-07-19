import os
import sys
import random

sys.path.append('../')
import open3d as o3d
import numpy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from readbnt import read_bntfile
random.seed(7122)

data_root = os.path.expanduser('~//yq_pointnet//BosphorusDB')

np.set_printoptions(suppress=True)
numpy.set_printoptions(suppress=True)
numpy.set_printoptions(threshold=numpy.inf)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=4)
numpy.set_printoptions(precision=4)

def rand_row(array, dim_needed):
    row_total = array.shape[0]
    row_sequence = np.arange(row_total)
    np.random.shuffle(row_sequence)
    return array[row_sequence[0:dim_needed], :]

class Bosphorus_Dataset(Dataset):
    """
    Args:
        csv_path (string): Path to the csv file with annotations.
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, csv_path, transform=None):
        csv_path = os.path.expanduser(csv_path)
        assert os.path.exists(csv_path), '%s not found' % csv_path

        # read csv file
        self.df = pd.read_csv(csv_path, header=0, names=['point_cloud_path', 'cls_name'])
        self.df.cls_name, _ = pd.factorize(self.df.cls_name, sort=True)

        self._num_of_classes = len(self.df.cls_name.drop_duplicates())
        self._len_of_dataset = len(self.df.cls_name)
        self._transform = transform

    def get_num_of_classes(self):
        return self._num_of_classes

    def __len__(self):
        return self._len_of_dataset

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        point_cloud_path,  cls_id = self.df.iloc[idx]
        if int(cls_id) > 104:
            point_cloud_data = np.loadtxt(point_cloud_path, delimiter=',')
            point_cloud_data = rand_row(point_cloud_data, 4000)
            point_cloud_data = point_cloud_data[:, 0:3]
        else:
            _, _, point_cloud_data = read_bntfile(point_cloud_path)

        #pcd = o3d.io.read_point_cloud(point_cloud_path)
        #pcd = o3d.geometry.PointCloud.uniform_down_sample(pcd, 4)
        #print("pcd；  ", pcd)

        if numpy.any(numpy.isnan(point_cloud_data)):
            # point_cloud_data = point_cloud_data[~np.isnan(point_cloud_data).any(axis=1), :]
            point_cloud_data[np.isnan(point_cloud_data)] = 0

        point_cloud_data = point_cloud_data - np.expand_dims(np.mean(point_cloud_data, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_cloud_data ** 2, axis = 1)),0)
        point_cloud_data = point_cloud_data / dist #scale

        # fname = os.path.join(data_root, "{}.txt".format(cls_id))
        # save_data = point_cloud_data
        # print(save_data)
        # numpy.savetxt(fname, save_data, fmt='%.04f')
        point_cloud_data = torch.from_numpy(point_cloud_data.astype(np.float))
        cls_id = torch.from_numpy(np.array([cls_id]).astype(np.int64))
        return point_cloud_data, cls_id
