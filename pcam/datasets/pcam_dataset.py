import torch
from torch.utils.data import Dataset


class PCAMDataset(Dataset):
    def __init__(self, root, phase, voxel_size, num_points):
       super(PCAMDataset, self).__init__() 
       self.root = root
       self.phase = phase
       self.voxel_size = voxel_size
       self.num_points = num_points

    def __len__(self):
        return len(self.files)
