"""
Most of the code in this file is taken from https://github.com/chrischoy/DeepGlobalRegistration/blob/master/dataloader/threedmatch_loader.py

This is dataloader used in [1-3] and we re-use it to train and test PCAM on the same dataset.

[1] Christopher Choy, Wei Dong, Vladlen Koltun. Deep Global Registration, CVPR, 2020.
[2] Christopher Choy, Jaesik Park, Vladlen Koltun. Fully Convolutional Geometric Features. ICCV, 2019.
[3] Christopher Choy, JunYoung Gwak, Silvio Savarese. 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks. CVPR, 2019.
"""

import torch
import numpy as np
import os
import random
from tqdm import tqdm
from pcam.datasets.pcam_dataset import PCAMDataset
from pcam.tool.transforms import sample_random_trans, apply_transform, sample_points, ground_truth_attention
from pcam.tool.file import read_trajectory
import open3d as o3d
import MinkowskiEngine as ME
import glob


class ThreeDMatchDataset(PCAMDataset):
    OVERLAP_RATIO = 0.3
    dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'split')
    DATA_FILES = {
        'train': os.path.join(dir_path, 'train_3dmatch.txt'),
        'val': os.path.join(dir_path, 'val_3dmatch.txt'),
        'test': os.path.join(dir_path, 'test_3dmatch.txt'),
    }
    def __init__(self,
                 root,
                 phase,
                 min_scale=0.8,
                 max_scale=1.2,
                 random_scale=False,
                 rotation_range=360,
                 voxel_size=0.05,
                 num_points=2048):
        super(ThreeDMatchDataset, self).__init__(root, phase, voxel_size, num_points)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.random_scale = random_scale
        self.phase = phase

        self.files = []
        self.randg = np.random.RandomState()
        self.rotation_range = rotation_range

        subset_names = open(self.DATA_FILES[phase]).read().split()
        if phase == "test":
            for sname in subset_names:
                traj_file = os.path.join(self.root, sname + '-evaluation/gt.log')
                assert os.path.exists(traj_file)
                traj = read_trajectory(traj_file)
                for ctraj in traj:
                    i = ctraj.metadata[0]
                    j = ctraj.metadata[1]
                    T_gt = ctraj.pose
                    self.files.append((sname, i, j, T_gt))
        else:
            for name in subset_names:
              fname = name + "*%.2f.txt" % self.OVERLAP_RATIO
              fnames_txt = glob.glob(root + "/" + fname)
              assert len(fnames_txt) > 0, f"Make sure that the path {root} has data {fname}"
              for fname_txt in fnames_txt:
                with open(fname_txt) as f:
                  content = f.readlines()
                fnames = [x.strip().split() for x in content]
                for fname in fnames:
                  self.files.append([fname[0], fname[1]])


    def __getitem__(self, idx):
        if self.phase == "test":
            file_name, i, j, T_gt = self.files[idx]
            ply_name0 = os.path.join(self.root, file_name, f'cloud_bin_{i}.ply')
            ply_name1 = os.path.join(self.root, file_name, f'cloud_bin_{j}.ply')
            xyz0 = o3d.io.read_point_cloud(ply_name0)
            xyz1 = o3d.io.read_point_cloud(ply_name1)
            xyz0 = np.asarray(xyz0.points)
            xyz1 = np.asarray(xyz1.points)

            ############ DUE TO BIAS AT TRAINING TIME WITH CENTERED POINT CLOUD
            xyz0_mean = xyz0.mean(0, keepdims=True)
            xyz1_mean = xyz1.mean(0, keepdims=True)
            xyz0 = xyz0 - xyz0_mean
            xyz1 = xyz1 - xyz1_mean
            ############

        else:
            file0 = os.path.join(self.root, self.files[idx][0])
            file1 = os.path.join(self.root, self.files[idx][1])
            data0 = np.load(file0)
            data1 = np.load(file1)
            file_name = self.files[idx][0] + "_" + self.files[idx][1]
            xyz0 = data0["pcd"]
            xyz1 = data1["pcd"]

            if self.random_scale and random.random() < 0.95:
                scale = self.min_scale + (self.max_scale - self.min_scale) * random.random()
                xyz0 = scale * xyz0
                xyz1 = scale * xyz1
            T0 = sample_random_trans(xyz0, self.randg, self.rotation_range)
            T1 = sample_random_trans(xyz1, self.randg, self.rotation_range)
            T_gt = T1 @ np.linalg.inv(T0)

            xyz0 = apply_transform(xyz0, T0)
            xyz1 = apply_transform(xyz1, T1)

            xyz0_mean, xyz1_mean = np.zeros((1, 3)), np.zeros((1, 3))

        # Voxelization
        xyz0_th = torch.from_numpy(xyz0)
        xyz1_th = torch.from_numpy(xyz1)

        sel0 = ME.utils.sparse_quantize(xyz0_th / self.voxel_size, return_index=True)
        sel1 = ME.utils.sparse_quantize(xyz1_th / self.voxel_size, return_index=True)
        
        unique_xyz0_th = xyz0_th[sel0]
        unique_xyz1_th = xyz1_th[sel1]
        
        unique_xyz0_th, unique_xyz1_th = unique_xyz0_th.float().numpy(), unique_xyz1_th.float().numpy()
        unique_xyz0_th = sample_points(unique_xyz0_th, self.num_points)
        unique_xyz1_th = sample_points(unique_xyz1_th, self.num_points)

        if self.phase == "test":
            T_gt = np.linalg.inv(T_gt)

        one_one_attention = ground_truth_attention(unique_xyz0_th, unique_xyz1_th, T_gt)

        return xyz0_th, xyz1_th, unique_xyz0_th, unique_xyz1_th, T_gt, np.linalg.inv(T_gt), one_one_attention.A, file_name, xyz0_mean, xyz1_mean


