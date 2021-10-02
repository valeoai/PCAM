"""
Most of the code in this file is taken from https://github.com/WangYueFt/prnet/blob/master/data.py

This is dataloader used in PrNet [1] that we re-use to train and test on the same dataset.
The modifications concern extra information needed to train and test PCAM.

[1] Yue Wang and Justin M. Solomon. PRNet: Self-Supervised Learning for Partial-to-Partial Registration. NeurIPS, 2019.
"""

import os
import sys
import glob
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski
import lightconvpoint.nn as lcp_nn
import torch
from pcam.tool.transforms import ground_truth_attention


# --- Base directory
basedir = os.path.dirname(os.path.realpath(__file__))
BASE_DIR = basedir + "/../data/modelnet/"


def load_data(partition):
    DATA_DIR = BASE_DIR
    all_data = []
    all_label = []

    print()
    print(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition))
    print()

    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name, 'r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def farthest_subsample_points(pointcloud1, pointcloud2, num_subsampled_points=768):
    pointcloud1 = pointcloud1.T
    pointcloud2 = pointcloud2.T
    num_points = pointcloud1.shape[0]
    nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud1)
    random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
    idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
    nbrs2 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud2)
    random_p2 = random_p1 #np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 2, -2])
#     random_p2 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 2, -2])
    idx2 = nbrs2.kneighbors(random_p2, return_distance=False).reshape((num_subsampled_points,))
    return pointcloud1[idx1, :].T, pointcloud2[idx2, :].T


class ModelNet40(Dataset):
    def __init__(self, num_points, num_subsampled_points=768, partition='train',
                 gaussian_noise=False, unseen=False, rot_factor=4, category=None):
        super(ModelNet40, self).__init__()
        if partition in ['train', 'val']:
            self.data, self.label = load_data('train')
        else:
            self.data, self.label = load_data('test')
            

        if category is not None:
            self.data = self.data[self.label==category]
            self.label = self.label[self.label==category]
        self.num_points = num_points
        self.num_subsampled_points = num_subsampled_points
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.unseen = unseen
        self.label = self.label.squeeze()
        self.rot_factor = rot_factor
        if num_points != num_subsampled_points:
            self.subsampled = True
        else:
            self.subsampled = False
        if self.unseen:
            ######## simulate testing on first 20 categories while training on last 20 categories
            if self.partition == 'test':
                self.data = self.data[self.label>=20]
                self.label = self.label[self.label>=20]
            elif self.partition == "val":
                self.data = self.data[(self.label>=16) & (self.label < 20)]
                self.label = self.label[(self.label>=16) & (self.label < 20)]
            elif self.partition == 'train':
                self.data = self.data[self.label<16]
                self.label = self.label[self.label<16]
        else:
            if partition == 'train' or partition == "val":
                np.random.seed(42)
                p = np.random.permutation(self.data.shape[0])
                self.data = self.data[p, :, :] 
                self.label = self.label[p]
                if partition == 'train':
                    self.data = self.data[:8000, :, :]
                    self.label = self.label[:8000] 
                elif partition == 'val':
                    self.data = self.data[8000:, :, :]
                    self.label = self.label[8000:]


    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        if self.partition != 'train':
            np.random.seed(item)
        anglex = np.random.uniform() * np.pi / self.rot_factor
        angley = np.random.uniform() * np.pi / self.rot_factor
        anglez = np.random.uniform() * np.pi / self.rot_factor
        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                        [0, cosx, -sinx],
                        [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                        [0, 1, 0],
                        [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                        [sinz, cosz, 0],
                        [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T
        translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                   np.random.uniform(-0.5, 0.5)])
        translation_ba = -R_ba.dot(translation_ab)

        pointcloud1 = pointcloud.T

        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)

        euler_ab = np.asarray([anglez, angley, anglex])
        euler_ba = -euler_ab[::-1]

        pointcloud1 = np.random.permutation(pointcloud1.T).T
        pointcloud2 = np.random.permutation(pointcloud2.T).T

        if self.gaussian_noise:
            pointcloud1 = jitter_pointcloud(pointcloud1)
            pointcloud2 = jitter_pointcloud(pointcloud2)

        if self.subsampled:
            pointcloud1, pointcloud2 = farthest_subsample_points(pointcloud1, pointcloud2,
                                                                 num_subsampled_points=self.num_subsampled_points)
        search = lcp_nn.SearchQuantized(K=32, stride=1)
        indices1, pointcloud1 = search(torch.from_numpy(pointcloud1).float().unsqueeze(0))
        indices2, pointcloud2 = search(torch.from_numpy(pointcloud2).float().unsqueeze(0))
        pointcloud1 = pointcloud1.squeeze().numpy()
        pointcloud2 = pointcloud2.squeeze().numpy()

        ###########
        # Ground truth attention matrix for training
        ret_trans = np.identity(4)
        ret_trans[:3, :3] = R_ab
        ret_trans[:3, 3] = translation_ab
        attention = ground_truth_attention(pointcloud1.T, pointcloud2.T, ret_trans)
        ###########

        return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
               translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
               euler_ab.astype('float32'), euler_ba.astype('float32'), \
               indices1.squeeze().numpy(), indices2.squeeze().numpy(), \
               attention.A, np.zeros((1, 3)), np.zeros((1, 3))

    def __len__(self):
        return self.data.shape[0]


def collate_fn(data):

    pointcloud1, pointcloud2, R_ab, translation_ab, \
    R_ba, translation_ba, euler_ab, euler_ba, \
    indices1, indices2, one_one_attention, p0_mean, p1_mean = list(zip(*data))

    pts1 = torch.from_numpy(np.stack(pointcloud1)).float()
    pts2 = torch.from_numpy(np.stack(pointcloud2)).float()
    raw_pts1 = torch.from_numpy(np.stack(pointcloud1)).float()
    raw_pts2 = torch.from_numpy(np.stack(pointcloud2)).float()
    orig_raw_pts1 = torch.from_numpy(np.stack(pointcloud1)).float()
    orig_raw_pts2 = torch.from_numpy(np.stack(pointcloud2)).float()

    p0_mean = torch.from_numpy(np.stack(p0_mean)).float().transpose(1, 2)
    p1_mean = torch.from_numpy(np.stack(p1_mean)).float().transpose(1, 2)

    indices1 = torch.from_numpy(np.stack(indices1)).float()
    indices2 = torch.from_numpy(np.stack(indices2)).float()

    Rs = torch.from_numpy(np.stack(R_ab)).float()
    ts = torch.from_numpy(np.stack(translation_ab)).float().unsqueeze(-1)
    Rs_inv = torch.from_numpy(np.stack(R_ba)).float()
    ts_inv = torch.from_numpy(np.stack(translation_ba)).float().unsqueeze(-1)

    one_one_attention = torch.from_numpy(np.stack(one_one_attention)).float()

    return {
        "raw_src": raw_pts1,
        "raw_tgt": raw_pts2,
        "src": pts1,
        "tgt": pts2,
        "attention": one_one_attention,
        "indices1": indices1,
        "indices2": indices2,
        "rotation": Rs,
        "translation": ts,
        "inv_rotation": Rs_inv,
        "inv_translation": ts_inv,
        "orig_raw_src": orig_raw_pts1,
        "orig_raw_tgt": orig_raw_pts2,
        "p0_mean": p0_mean,
        "p1_mean": p1_mean,
    }

