from pcam.datasets.threedmatch_dataset import ThreeDMatchDataset
from torch.utils.data import DataLoader
from pcam.datasets.collate import CollateFunc


class ThreeDMatchDataModule():
    def __init__(self, root, num_points):
        self.root = root
        self.num_points = num_points
        self.collate_fn = CollateFunc()

    def train_loader(self):
        self.train_set = ThreeDMatchDataset(self.root, "train", num_points=self.num_points)
        return DataLoader(self.train_set,
                          pin_memory=True,
                          batch_size=1,
                          collate_fn=self.collate_fn,
                          num_workers=4,
                          shuffle=True)

    def val_loader(self):
        self.val_set = ThreeDMatchDataset(self.root, "val", num_points=self.num_points)
        return DataLoader(self.val_set,
                          pin_memory=True,
                          batch_size=1,
                          collate_fn=self.collate_fn,
                          num_workers=4,
                          shuffle=False)

    def test_loader(self):
        self.test_set = ThreeDMatchDataset(self.root, "test",
                                           min_scale=1., max_scale=1.,
                                           rotation_range=0,
                                           num_points=self.num_points)
        return DataLoader(self.test_set,
                          pin_memory=True,
                          batch_size=1,
                          collate_fn=self.collate_fn,
                          num_workers=4,
                          shuffle=False)
