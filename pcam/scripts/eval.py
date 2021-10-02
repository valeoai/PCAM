import os
import sys
import torch
import numpy as np
from types import SimpleNamespace
import pcam.datasets.modelnet40 as m40
from pcam.tool.test_final import test
from pcam.models.network import PoseEstimator
from pcam.datasets.kitti_dm import KittiDataModule
from pcam.datasets.threedmatch_dm import ThreeDMatchDataModule

from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from sacred import SETTINGS
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds


# Sacred config.
SETTINGS.CAPTURE_MODE = 'sys'  # for tqdm
ex = Experiment('eval_pcam')
ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def config():
    # Name of the dataset
    DATASET = ''
    # Number of points to sample for the point cloud
    NUM_POINTS = 2048
    # Use a sparse mapping or not when computing pairs of corresponding point
    SPARSE_ATTENTION = False
    # Filter out pairs of matching point with a low confidence score (below THRESHOLD)
    THRESHOLD = None
    # Refinement via the optimisation proposed in DGR (Deep Global Registration)
    DGR_OPTIM = False
    # Safeguard registration (as in DGR) when the confidence scores are below a threshold (WSUM_THRESHOLD)
    SAFEGUARD = False
    WSUM_THRESHOLD = None
    # Refinement via ICP
    ICP = False
    # Split on which to evaluate
    PHASE = 'test'
    # Add a prefix for saving the model
    PREFIX = ''
    # Device for torch
    DEVICE = 'cuda'
    # Numbers of layers in our encoder to find matching points
    NB_ENCODERS = 6
    # Set to False for PCAM. When set to True only the last attention is used to find corresponding point.
    LAST_ATTENTION = False
    # (De)activate losses during training
    LOSS_ENTROPY_ATTENTION = True
    LOSS_DIST_ATTENTION = False
    LOSS_DIST_CONFIDENCE = True


@ex.automain
def main(NUM_POINTS, DEVICE, THRESHOLD,
         SPARSE_ATTENTION, LAST_ATTENTION, WSUM_THRESHOLD, NB_ENCODERS, DATASET, PHASE,
         LOSS_ENTROPY_ATTENTION, LOSS_DIST_ATTENTION, LOSS_DIST_CONFIDENCE,
         PREFIX, ICP, SAFEGUARD, DGR_OPTIM):

    assert PHASE in ['val', 'test'], "PHASE must have value of either 'val' or 'test'"
    assert DATASET in ['kitti', '3dmatch', 'modelnet', 'modelnet_unseen', 'modelnet_noise'], "Wrong value for DATASET"

    # --- Base directory
    basedir = os.path.dirname(os.path.realpath(__file__)) + '/../'

    # --- Load dataset
    if DATASET == 'kitti':
        path2data = basedir + "/data/kitti/dataset"
        icp_cache_path = basedir + "/data/kitti/icp"
        config = SimpleNamespace(
            voxel_size=0.3,
            min_scale=1.,
            max_scale=1.,
        )
        threshold_rte = 0.6
        threshold_rre = 5
        num_points = NUM_POINTS
        voxel_size = config.voxel_size
        data_module = KittiDataModule(path2data, icp_cache_path, NUM_POINTS)
    elif DATASET == '3dmatch':
        if PHASE == "val":
            path2data = basedir + "/data/3dmatch/threedmatch"
            config = SimpleNamespace(
                voxel_size=0.05,
                min_scale=0.8,
                max_scale=1.2,
            )
            threshold_rte = 0.3
            threshold_rre = 15
            num_points = NUM_POINTS
            voxel_size = config.voxel_size
            data_module = ThreeDMatchDataModule(path2data, NUM_POINTS)
        elif PHASE == 'test':
            path2data = basedir + "/data/3dmatch/threedmatch_test"
            config = SimpleNamespace(
                voxel_size=0.05,
                min_scale=1.,
                max_scale=1.,
            )
            threshold_rte = 0.3
            threshold_rre = 15
            num_points = NUM_POINTS
            voxel_size = config.voxel_size
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
        partition = PHASE

        loader = torch.utils.data.DataLoader(
            m40.ModelNet40(1024, partition=partition, gaussian_noise=gaussian_noise, unseen=unseen),
            pin_memory=True,
            batch_size=1,
            collate_fn=m40.collate_fn,
            num_workers=4,
            shuffle=False
        )
        threshold_rte = None
        threshold_rre = None
        num_points = None
        voxel_size = None
    else:
        raise NotImplementedError('Dataset not available')

    # val or test
    if DATASET == 'kitti' or DATASET == '3dmatch':
        if PHASE == "val":
            loader = data_module.val_loader()
        elif PHASE == "test":
            loader = data_module.test_loader()

    # --- Network
    BACKPROP = not SPARSE_ATTENTION
    net = PoseEstimator(
        nb_encoders=NB_ENCODERS,
        last_attention=LAST_ATTENTION,
        sparse_attention=SPARSE_ATTENTION,
        backprop=BACKPROP,
        threshold=THRESHOLD,
        N=None,
    ).to(DEVICE)

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
    print(path2exp)

    # --- Reload model
    print('Reload last checkpoint')
    checkpoint = torch.load(os.path.join(path2exp, 'check_point_last.pth'))
    net.load_state_dict(checkpoint['net'])
    recall_best = checkpoint['best_recall'] if checkpoint.get('best_recall') is not None else 0
    rre_best = checkpoint['best_rre'] if checkpoint.get('best_rre') is not None else np.inf
    print('Result from training:', checkpoint['epoch'], recall_best, rre_best)

    # ---
    test(loader, net, DEVICE,
         threshold_rte=threshold_rte,
         threshold_rre=threshold_rre,
         num_points=num_points,
         voxel_size=voxel_size,
         wsum_threshold=WSUM_THRESHOLD,
         icp=ICP,
         safeguard=SAFEGUARD,
         dgr_optim=DGR_OPTIM,
         dataset=DATASET,
         nb_average=1)
