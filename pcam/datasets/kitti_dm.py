# Part of the code in this file is taken from https://github.com/chrischoy/DeepGlobalRegistration/blob/master/dataloader/kitti_loader.py

from pcam.datasets.kitti_dataset import KittiDataset
from torch.utils.data import DataLoader
from pcam.datasets.collate import CollateFunc


class KittiDataModule():
    def __init__(self, root, icp_path, num_points):
        self.train_set = KittiDataset(root, "train", icp_path, num_points=num_points)
        self.val_set = KittiDataset(root, "val", icp_path, num_points=num_points)
        self.test_set = KittiDataset(root, "test", icp_path, num_points=num_points)
        self.collate_fn = CollateFunc()

    def train_loader(self):
        return DataLoader(self.train_set, 
                          pin_memory=True, 
                          batch_size=1, 
                          collate_fn=self.collate_fn,
                          num_workers=4, 
                          shuffle=True)

    def val_loader(self):
        return DataLoader(self.val_set,
                          pin_memory=True,
                          batch_size=1,
                          collate_fn=self.collate_fn,
                          num_workers=4,
                          shuffle=False)

    def test_loader(self):
        return DataLoader(self.test_set,
                          pin_memory=True,
                          batch_size=1,
                          collate_fn=self.collate_fn,
                          num_workers=4,
                          shuffle=False)
