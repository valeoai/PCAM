"""
Most of the code in this file is taken from https://github.com/chrischoy/DeepGlobalRegistration/blob/master/dataloader/threedmatch_loader.py

This is dataloader used in [1-3] and we re-use it to train and test PCAM on the same dataset.

[1] Christopher Choy, Wei Dong, Vladlen Koltun. Deep Global Registration, CVPR, 2020.
[2] Christopher Choy, Jaesik Park, Vladlen Koltun. Fully Convolutional Geometric Features. ICCV, 2019.
[3] Christopher Choy, JunYoung Gwak, Silvio Savarese. 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks. CVPR, 2019.
"""

import glob
import os
import numpy as np
import torch
from pcam.datasets.pcam_dataset import PCAMDataset
import MinkowskiEngine as ME
from pcam.tool.pointcloud import make_open3d_point_cloud
from pcam.tool.transforms import apply_transform, sample_points, ground_truth_attention
import open3d as o3d
from tqdm import tqdm


class KittiDataset(PCAMDataset):
    dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'split')
    DATA_FILES = {
        'train': os.path.join(dir_path, 'train_kitti.txt'),
        'val': os.path.join(dir_path, 'val_kitti.txt'),
        'test': os.path.join(dir_path, 'test_kitti.txt'),
    }
    MIN_DIST = 10
    def __init__(self,
                 root,
                 phase,
                 icp_path,
                 voxel_size=0.3,
                 num_points=4096):
        super(KittiDataset, self).__init__(root, phase, voxel_size, num_points)
        self.icp_path = icp_path
        max_time_diff = 3

        self.files = []
        self.kitti_icp_cache = {}
        self.kitti_cache = {}

        subset_names = open(self.DATA_FILES[phase]).read().split()
        for dirname in subset_names:
            drive_id = int(dirname)
            dirname = root + '/sequences/%02d/velodyne/*.bin' % drive_id
            print(dirname)
            fnames = glob.glob(dirname)
            assert len(fnames) > 0, f"Make sure that the path {root} has data {dirname}"
            inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

            all_odo = self.get_video_odometry(drive_id, return_all=True)
            all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
            Ts = all_pos[:, :3, 3]
            pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3))**2
            pdist = np.sqrt(pdist.sum(-1))
            more_than_10 = pdist > self.MIN_DIST
            curr_time = inames[0]
            while curr_time in inames:
                # Find the min index
                next_time = np.where(more_than_10[curr_time][curr_time:curr_time + 100])[0]
                if len(next_time) == 0:
                    curr_time += 1
                else:
                    # Follow https://github.com/yewzijian/3DFeatNet/blob/master/scripts_data_processing/kitti/process_kitti_data.m#L44
                    next_time = next_time[0] + curr_time - 1

                if next_time in inames:
                    self.files.append((drive_id, curr_time, next_time))
                    if phase == "train":
                        #             curr_time += 5
                        curr_time = next_time + 1
                    else:
                        curr_time = next_time + 1

        # Remove problematic sequence
        for item in [
            (8, 15, 58),
        ]:
            if item in self.files:
                self.files.pop(self.files.index(item))


    def __getitem__(self, idx):
        drive = self.files[idx][0]
        t0, t1 = self.files[idx][1], self.files[idx][2]
        key = '%d_%d_%d' % (drive, t0, t1)
        filename = self.icp_path + '/' + key + '.npy'
        pts1_file = self.icp_path + '/' + key + '_pts1.npy'
        pts2_file = self.icp_path + '/' + key + '_pts2.npy'

        fname0 = self._get_velodyne_fn(drive, t0)
        fname1 = self._get_velodyne_fn(drive, t1)

        if key not in self.kitti_icp_cache:
            if not os.path.exists(filename):
                all_odometry = self.get_video_odometry(drive, [t0, t1])
                positions = [self.odometry_to_positions(odometry) for odometry in all_odometry]

                # XYZ and reflectance
                xyzr0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)
                xyzr1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)

                xyz0 = xyzr0[:, :3]
                xyz1 = xyzr1[:, :3]

                # work on the downsampled xyzs, 0.05m == 5cm
                sel0 = ME.utils.sparse_quantize(xyz0 / 0.05, return_index=True)
                sel1 = ME.utils.sparse_quantize(xyz1 / 0.05, return_index=True)

                M = (self.velo2cam @ positions[0].T @ np.linalg.inv(positions[1].T)
                     @ np.linalg.inv(self.velo2cam)).T
                xyz0_t = apply_transform(xyz0[sel0], M)
                pcd0 = make_open3d_point_cloud(xyz0_t)
                pcd1 = make_open3d_point_cloud(xyz1[sel1])
                reg = o3d.registration.registration_icp(pcd0, pcd1, 0.2, np.eye(4),
                                                        o3d.registration.TransformationEstimationPointToPoint(),
                                                        o3d.registration.ICPConvergenceCriteria(max_iteration=200))
                pcd0.transform(reg.transformation)
                T_gt = M @ reg.transformation
                np.save(filename, T_gt)
            else:
                T_gt = np.load(filename)
            self.kitti_icp_cache[key] = T_gt
        else:
            T_gt = self.kitti_icp_cache[key]

        if not os.path.exists(pts1_file):
            xyzr0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)
            xyzr1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)

            xyz0 = xyzr0[:, :3]
            xyz1 = xyzr1[:, :3]

            np.save(pts1_file, xyz0)
            np.save(pts2_file, xyz1)

        else:
            xyz0 = np.load(pts1_file)
            xyz1 = np.load(pts2_file)

        xyz0_full = torch.from_numpy(xyz0)
        xyz1_full = torch.from_numpy(xyz1)

        xyz0_th = xyz0_full
        xyz1_th = xyz1_full

        sel0 = ME.utils.sparse_quantize(xyz0_th / self.voxel_size, return_index=True)
        sel1 = ME.utils.sparse_quantize(xyz1_th / self.voxel_size, return_index=True)

        unique_xyz0_th = xyz0_th[sel0]
        unique_xyz1_th = xyz1_th[sel1]

        unique_xyz0_th, unique_xyz1_th = unique_xyz0_th.float().numpy(), unique_xyz1_th.float().numpy()
        unique_xyz0_th = sample_points(unique_xyz0_th, self.num_points)
        unique_xyz1_th = sample_points(unique_xyz1_th, self.num_points)

        attention = ground_truth_attention(unique_xyz0_th, unique_xyz1_th, T_gt)
        xyz0_mean, xyz1_mean = np.zeros((1, 3)), np.zeros((1, 3))

        return xyz0_th, xyz1_th, unique_xyz0_th, unique_xyz1_th, T_gt, np.linalg.inv(T_gt), attention.A, filename, xyz0_mean, xyz1_mean


    def get_all_scan_ids(self, drive_id):
        fnames = glob.glob(self.root + '/sequences/%02d/velodyne/*.bin' % drive_id)
        assert len(
            fnames) > 0, f"Make sure that the path {self.root} has drive id: {drive_id}"
        inames = [int(os.path.split(fname)[-1][:-4]) for fname in fnames]
        return inames

    @property
    def velo2cam(self):
        try:
            velo2cam = self._velo2cam
        except AttributeError:
            R = np.array([
                7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
                -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
            ]).reshape(3, 3)
            T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
            velo2cam = np.hstack([R, T])
            self._velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
        return self._velo2cam


    def _get_velodyne_fn(self, drive, t):
        fname = self.root + '/sequences/%02d/velodyne/%06d.bin' % (drive, t)
        return fname

    def odometry_to_positions(self, odometry):
        T_w_cam0 = odometry.reshape(3, 4)
        T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
        return T_w_cam0

    def get_video_odometry(self, drive, indices=None, ext='.txt', return_all=False):
        data_path = self.root + '/poses/%02d.txt' % drive
        if data_path not in self.kitti_cache:
            self.kitti_cache[data_path] = np.genfromtxt(data_path)
        if return_all:
            return self.kitti_cache[data_path]
        else:
            return self.kitti_cache[data_path][indices]
