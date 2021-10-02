import os
import sys
import torch
import tarfile
import numpy as np
from types import SimpleNamespace
import pcam.datasets.modelnet40 as m40
from pcam.models.network import PoseEstimator
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from pcam.tool.train_util import train_one_epoch, validation
import math
from pcam.datasets.kitti_dm import KittiDataModule
from pcam.datasets.threedmatch_dm import ThreeDMatchDataModule

from sacred import Experiment
from sacred import SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import MongoObserver


# Sacred config.
SETTINGS.CAPTURE_MODE = 'sys'  # for tqdm
name = 'pcam_train'
ex = Experiment(name)


@ex.config
def config():
    # Name of the dataset
    DATASET = ''
    # Number of points to sample for the point cloud
    NUM_POINTS = 2048
    # Device for torch
    DEVICE = 'cuda'
    # Name of subfolder where to save checkpoint
    PREFIX = 'new_training'
    # Use a sparse mapping or not when computing pairs of corresponding point
    SPARSE_ATTENTION = True
    # Set to False for PCAM. When set to True only the last attention is used to find corresponding point.
    LAST_ATTENTION = False
    # Numbers of layers in our encoder to find matching points
    NB_ENCODERS = 6
    # Number of epochs
    NB_EPOCHS = 100
    # In case training needs to be relaunched from last checkpoint
    RELOAD_LAST = False
    # (De)activate losses during training
    LOSS_ENTROPY_ATTENTION = True
    LOSS_DIST_ATTENTION = False
    LOSS_DIST_CONFIDENCE = True


def save_model(epoch, net, optimizer, scheduler, best_recall, best_rre, filename):

    torch.save(
        {
            'epoch': epoch,
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_recall': best_recall,
            'best_rre': best_rre,
        },
        filename
    )


@ex.automain
def main(NB_EPOCHS, NUM_POINTS, RELOAD_LAST, DEVICE, PREFIX,
         SPARSE_ATTENTION, LAST_ATTENTION, NB_ENCODERS, DATASET,
         LOSS_ENTROPY_ATTENTION, LOSS_DIST_ATTENTION, LOSS_DIST_CONFIDENCE):

    # --- Base directory
    basedir = os.path.dirname(os.path.realpath(__file__)) + '/../'

    # --- Load dataset
    if DATASET == 'kitti':
        path2data = basedir + "/data/kitti/dataset"
        icp_cache_path = basedir + "/data/kitti/icp"
        threshold_rte = 0.6
        threshold_rre = 5
        config = SimpleNamespace(
            voxel_size=0.3,
            min_scale=1.,
            max_scale=1.,
        )
        data_module = KittiDataModule(path2data, icp_cache_path, NUM_POINTS)
    elif DATASET == '3dmatch':
        path2data = basedir + "/data/3dmatch/threedmatch"
        config = SimpleNamespace(
            voxel_size=0.05,
            min_scale=0.8,
            max_scale=1.2,
        )
        threshold_rte = 0.3
        threshold_rre = 15
        data_module = ThreeDMatchDataModule(path2data, NUM_POINTS)
    elif DATASET[:8] == 'modelnet':
        unseen = False
        gaussian_noise = False
        if len(DATASET) > 8:
            if DATASET[9:] == 'unseen':
                unseen = True
            elif DATASET[9:] == 'noise':
                gaussian_noise = True
            else:
                raise NotImplementedError('Dataset not available')
        print('unseen', unseen, 'gaussian_noise', gaussian_noise)
        train_loader = torch.utils.data.DataLoader(
            m40.ModelNet40(1024, partition='train', gaussian_noise=gaussian_noise, unseen=unseen),
            pin_memory=True,
            batch_size=1,
            collate_fn=m40.collate_fn,
            num_workers=4,
            shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            m40.ModelNet40(1024, partition='val', gaussian_noise=gaussian_noise, unseen=unseen),
            pin_memory=True,
            batch_size=1,
            collate_fn=m40.collate_fn,
            num_workers=4,
            shuffle=False
        )
        # For validation accuracy during training
        threshold_rte = 0.03
        threshold_rre = 1.
        # Only used to define threshold for good pairs of points
        config = SimpleNamespace(
            voxel_size=0.03,
            max_scale=1.,
        )

    if DATASET == 'kitti' or DATASET == '3dmatch':
        train_loader = data_module.train_loader()
        val_loader = data_module.val_loader()

    # --- Network
    BACKPROP = not SPARSE_ATTENTION
    net = PoseEstimator(
        nb_encoders=NB_ENCODERS,
        last_attention=LAST_ATTENTION,
        sparse_attention=SPARSE_ATTENTION,
        backprop=BACKPROP
    ).to(DEVICE)

    # --- Optimizer
    epoch_factor = NB_EPOCHS / 100.0
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = MultiStepLR(optimizer,
                            milestones=[int(60 * epoch_factor), int(80 * epoch_factor)],
                            gamma=0.1)

    # --- Path to experiment
    root = basedir + "/trained_models/"
    path2exp = root + DATASET + '/' + PREFIX + '/'
    if SPARSE_ATTENTION:
        path2exp += 'sparse_'
    else:
        path2exp += 'soft_'
    if LAST_ATTENTION:
        path2exp += 'lastAttention_'
    path2exp += 'nbEnc_' + str(NB_ENCODERS)
    if not BACKPROP:
        path2exp += '_noBackprop'
    if LOSS_DIST_ATTENTION:
        if LOSS_ENTROPY_ATTENTION:
            path2exp += '_DistEntAtt'
        else:
            path2exp += '_DistAtt'
    else:
        if not LOSS_ENTROPY_ATTENTION:
            path2exp += '_NoLossAtt'
    if LOSS_DIST_CONFIDENCE:
        path2exp += '_DistConf'
    if NUM_POINTS != 2048:
        path2exp += '_' + str(NUM_POINTS)
    print('Save experiment in ' + path2exp)

    # --- Reload model
    if RELOAD_LAST:
        print('Reload last checkpoint')
        checkpoint = torch.load(os.path.join(path2exp, 'check_point_last.pth'))
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch = checkpoint['epoch'] + 1
        purge_step = epoch * len(train_loader)
        recall_best = checkpoint['best_recall'] if checkpoint.get('best_recall') is not None else 0
        rre_best = checkpoint['best_rre'] if checkpoint.get('best_rre') is not None else np.inf
    else:
        epoch = 0
        purge_step = 0
        recall_best = 0
        rre_best = np.inf

    # --- Tensorboard
    logger = SummaryWriter(log_dir=path2exp, purge_step=purge_step, flush_secs=60)

    # --- Train
    for epoch in range(epoch, NB_EPOCHS):

        train_one_epoch(epoch, train_loader, net, optimizer, logger, DEVICE,
                        2 * config.max_scale * config.voxel_size,
                        LOSS_ENTROPY_ATTENTION, LOSS_DIST_ATTENTION, LOSS_DIST_CONFIDENCE)
        recall, rre = validation(epoch, val_loader, net, logger, DEVICE,
                                 threshold_rte=threshold_rte,
                                 threshold_rre=threshold_rre)
        scheduler.step()

        # Save best model (but we use the last saved model at test time !)
        if recall_best < recall or (recall_best == recall and rre_best > rre):
            recall_best, rre_best = recall, rre
            filename = os.path.join(path2exp, 'check_point_best.pth')
            save_model(epoch, net, optimizer, scheduler, recall_best, rre_best, filename)

        # Save checkpoint
        print(epoch + 1, 'epoch done')
        filename = os.path.join(path2exp, 'check_point_last.pth')
        save_model(epoch, net, optimizer, scheduler, recall_best, rre_best, filename)
