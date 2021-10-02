# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
import torch
import numpy as np
import random
from scipy.linalg import expm, norm
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from .pointcloud import get_matching_indices, make_open3d_point_cloud


def decompose_rotation_translation(Ts):
  Ts = Ts.float()
  Rs = Ts[:, :3, :3]
  ts = Ts[:, :3, 3]

  Rs.require_grad = False
  ts.require_grad = False

  return Rs, ts

def voxelize(point_cloud, voxel_size):
    # Random permutation (for random selection within voxel)
    point_cloud = np.random.permutation(point_cloud)

    # Set minimum value to 0 on each axis
    min_val = point_cloud.min(0)
    pc = point_cloud - min_val

    # Quantize
    pc = np.floor(pc / voxel_size)
    L, M, N = pc.max(0) + 1
    pc = pc[:, 0] + L * pc[:, 1] + L * M * pc[:, 2]

    # Select voxel
    _, idx = np.unique(pc, return_index=True)

    return point_cloud[idx, :]

def sample_points(pts, num_points):
    if pts.shape[0] > num_points:
        pts = np.random.permutation(pts)[:num_points]
    else:
        pts = np.random.permutation(pts)
    return pts

def ground_truth_attention_distance(xyz0, xyz1, trans, search_voxel_size):
    
    pcd0 = make_open3d_point_cloud(xyz0)
    pcd1 = make_open3d_point_cloud(xyz1)
    # Sizes
    N = xyz0.shape[0]
    M = xyz1.shape[0]
    matches = get_matching_indices(pcd0, pcd1, trans, search_voxel_size, K=None)
    
    l, r = [], []
    for i, j in matches:
        l.append(i)
        r.append(j)
    
    A = csr_matrix((np.ones(len(matches)), (l, r)), (N, M))
#     print(A.shape)
#     print(A.sum(axis=1).shape)
#     A = A / A.sum(axis=1)
#     B = csr_matrix((distance[:, 0], (np.arange(M), neighbors)), (M, N)).T
#     A = A * B

    return A

def ground_truth_attention(p1, p2, trans):
    
    # Ideal pts2 with ground truth transformation
    ideal_pts2 = p1 @ trans[:3, :3].T + trans[:3, 3:4].T
    
    # Sizes
    N = p1.shape[0]
    M = p2.shape[0]
    
    # Search NN for each ideal_pt2 in p2
    nn = NearestNeighbors(n_neighbors=1).fit(p2)
    distance, neighbors = nn.kneighbors(ideal_pts2)
    neighbors = neighbors[:, 0]
#     print(neighbors, len(neighbors))
    
    # Create ideal attention matrix
#     A = csr_matrix((distance[:, 0], (np.arange(N), neighbors)), (N, M))
    A = csr_matrix((np.ones(N), (np.arange(N), neighbors)), (N, M))
#     print(A.shape)

    # Search NN for each p2 in ideal_pt2
    nn = NearestNeighbors(n_neighbors=1).fit(ideal_pts2)
    distance, neighbors = nn.kneighbors(p2)
    neighbors = neighbors[:, 0]
    
    # Create ideal attention matrix
    B = csr_matrix((np.ones(M), (np.arange(M), neighbors)), (M, N)).T
#     B = csr_matrix((distance[:, 0], (np.arange(M), neighbors)), (M, N)).T
    
    # Keep only consistent neighbors by pointwise multiplication
#     thres = 0.03
#     A = A.toarray()
#     B = B.toarray()
#     A = (A < thres) & (A > 0)
#     B = (B < thres) & (B > 0)
    A = A.multiply(B) 
    
#     A = A * B
    return A

# Rotation matrix along axis with angle theta
def M(axis, theta):
    return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

def M_z_axis(theta):
    theta = theta * np.pi / 180.0
    z_axis = np.array([0, 0, 1])
    return expm(np.cross(np.eye(3), z_axis / norm(z_axis) * theta))

def sample_random_rotation_z_axis(pcd, randg, rotation_range=360):
    T = np.eye(4)
    z_axis = np.array([0, 0, 1])    
    random_angle = rotation_range * (randg.rand(1) - 0.5)
    R = M(z_axis, random_angle * np.pi / 180.0)
    T[:3, :3] = R
    T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
    return T

def sample_random_trans(pcd, randg, rotation_range=360):
  T = np.eye(4)
  R = M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
  T[:3, :3] = R  
  T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
  return T

def apply_transform(pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    pts = pts @ R.T + T
    return pts

class Compose:
  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, coords, feats):
    for transform in self.transforms:
      coords, feats = transform(coords, feats)
    return coords, feats


class Jitter:
  def __init__(self, mu=0, sigma=0.01):
    self.mu = mu
    self.sigma = sigma

  def __call__(self, coords, feats):
    if random.random() < 0.95:
      feats += self.sigma * torch.randn(feats.shape[0], feats.shape[1])
      if self.mu != 0:
        feats += self.mu
    return coords, feats


class ChromaticShift:
  def __init__(self, mu=0, sigma=0.1):
    self.mu = mu
    self.sigma = sigma

  def __call__(self, coords, feats):
    if random.random() < 0.95:
      feats[:, :3] += torch.randn(self.mu, self.sigma, (1, 3))
    return coords, feats
