import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from pcam.tool.log import Logger, save_metrics


@torch.no_grad()
def validation(epoch, val_loader, net, tensorboard, device, threshold_rte, threshold_rre):
    """
    """

    # Init.
    net = net.eval()

    #
    logger = Logger()
    mean_epe_true_mask = 0
    mean_epe_detect_mask = 0

    #
    for it, batch in enumerate(tqdm(val_loader)):

        # Extract data
        src = batch['src'].float().to(device)
        tgt = batch['tgt'].float().to(device)
        src_indices = batch['indices1'].long().to(device)
        tgt_indices = batch['indices2'].long().to(device)
        true_attention = batch['attention'].float().to(device)
        R = batch['rotation'].float().to(device)
        T = batch['translation'].float().to(device)

        # Match points
        Rest, Test, corres_pts_for_src, _, _, _, w_src, _ = \
            net(src, tgt, indices1=src_indices, indices2=tgt_indices)

        # Save metric
        save_metrics(logger, 'val', R, T, Rest, Test, threshold_rte, threshold_rre)

        # Log
        est_mask_src = w_src > 0.5
        true_mask_src = true_attention.sum(2, keepdim=True).transpose(1, 2) > 0
        gt_corres_pts_for_src = torch.bmm(R, src) + T
        epe = torch.norm(corres_pts_for_src - gt_corres_pts_for_src, p=2, dim=1, keepdim=True)
        mean_epe_true_mask += epe[true_mask_src].mean().item()
        mean_epe_detect_mask += epe[est_mask_src].mean().item()

    # Log
    tensorboard.add_scalar('Acc_Val/recall', np.mean(logger.store['val.recall']), epoch)
    tensorboard.add_scalar('Acc_Val/rte', np.mean(logger.store['val.rte']), epoch)
    tensorboard.add_scalar('Acc_Val/rre', np.mean(logger.store['val.rre']), epoch)
    tensorboard.add_scalar('Acc_Val/rte_all', np.mean(logger.store['val.rte_all']), epoch)
    tensorboard.add_scalar('Acc_Val/rre_all', np.mean(logger.store['val.rre_all']), epoch)
    tensorboard.add_scalar('EPE_Val/true_mask', mean_epe_true_mask / (it + 1), epoch)
    tensorboard.add_scalar('EPE_Val/detected_mask', mean_epe_detect_mask / (it + 1), epoch)

    return np.mean(logger.store['val.recall']), np.mean(logger.store['val.rre_all'])


def train_one_epoch(epoch, train_loader, net, optimizer, tensorboard, device, threshold,
                    loss_entropy_attention, loss_dist_attention, loss_dist_confidence):
    """
    """

    # Init.
    net = net.train()
    shift = epoch * len(train_loader)

    #
    delta = 100
    mean_lea = 0
    mean_lda = 0
    mean_lec = 0
    mean_ldc = 0
    mean_loss = 0
    mean_found = 0
    mean_correct = 0
    mean_good_pairs = 0
    zero = torch.Tensor([0.]).to(device)

    #
    for it, batch in enumerate(tqdm(train_loader)):

        # Extract data
        src = batch['src'].float().to(device)
        tgt = batch['tgt'].float().to(device)
        src_indices = batch['indices1'].long().to(device)
        tgt_indices = batch['indices2'].long().to(device)
        true_attention = batch['attention'].float().to(device)
        R = batch['rotation'].float().to(device)
        T = batch['translation'].float().to(device)
        Rinv = batch['inv_rotation'].float().to(device)
        Tinv = batch['inv_translation'].float().to(device)

        # Match points
        _, _, corres_pts_for_src, corres_pts_for_tgt, log_attn_row, log_attn_col, w_src, w_tgt = \
            net(src, tgt, indices1=src_indices, indices2=tgt_indices)

        # --- Losses for point matcher
        # Precomputations
        gt_corres_pts_for_src = torch.bmm(R, src) + T
        gt_corres_pts_for_tgt = torch.bmm(Rinv, tgt) + Tinv
        # Cross entropy attention
        if loss_entropy_attention:
            lea = - (true_attention * log_attn_row).sum(2).mean()
            lea += - (true_attention * log_attn_col).sum(1).mean()
        else:
            lea = zero
        # EPE loss
        if loss_dist_attention:
            true_mask_src = true_attention.sum(2) > 0
            lda = torch.norm(corres_pts_for_src - gt_corres_pts_for_src, p=2, dim=1)[true_mask_src].mean()
            true_mask_tgt = true_attention.sum(1) > 0
            lda += torch.norm(corres_pts_for_tgt - gt_corres_pts_for_tgt, p=2, dim=1)[true_mask_tgt].mean()
        else:
            lda = zero

        # --- Losses for confidence estimator
        # Detach if no backprop from confidence estimator loss to attention
        if not net.backprop:
            corres_pts_for_src = corres_pts_for_src.detach()
            corres_pts_for_tgt = corres_pts_for_tgt.detach()
        # Cross entropy on confidence weights
        # Source side
        true_label = (torch.norm(corres_pts_for_src - gt_corres_pts_for_src, p=2, dim=1, keepdim=True) < threshold).float()
        lec = F.binary_cross_entropy(w_src, true_label)
        # Target side
        true_label = (torch.norm(corres_pts_for_tgt - gt_corres_pts_for_tgt, p=2, dim=1, keepdim=True) < threshold).float()
        lec += F.binary_cross_entropy(w_tgt, true_label)
        # EPE loss
        if loss_dist_confidence:
            # Source side
            ldc = (w_src * torch.norm(corres_pts_for_src - gt_corres_pts_for_src, p=2, dim=1, keepdim=True)).mean()
            # Target side
            ldc += (w_tgt * torch.norm(corres_pts_for_tgt - gt_corres_pts_for_tgt, p=2, dim=1, keepdim=True)).mean()
        else:
            ldc = zero

        # Total loss
        loss = lea + lda + lec + ldc

        # Gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log
        mean_lea += lea.item()
        mean_lda += lda.item()
        mean_lec += lec.item()
        mean_ldc += ldc.item()
        mean_loss += loss.item()
        mean_found += (w_tgt > .5).float().sum(-1).mean()
        mean_correct += ((w_tgt > .5).float() * true_label).sum(-1).mean()
        mean_good_pairs += true_label.sum(-1).mean()
        if it % delta == delta - 1:
            # Loss
            tensorboard.add_scalar('Loss_Train/loss', mean_loss / delta, shift + it)
            tensorboard.add_scalar('Loss_Train/entropy_att', mean_lea / delta, shift + it)
            tensorboard.add_scalar('Loss_Train/dist_att', mean_lda / delta, shift + it)
            tensorboard.add_scalar('Loss_Train/entropy_conf', mean_lec / delta, shift + it)
            tensorboard.add_scalar('Loss_Train/dist_conf', mean_ldc / delta, shift + it)
            # Accuracies
            tensorboard.add_scalar('Acc_Train/nb_to_found', mean_good_pairs / delta, shift + it)
            tensorboard.add_scalar('Acc_Train/nb_correct', mean_correct / delta, shift + it)
            tensorboard.add_scalar('Acc_Train/nb_found', mean_found / delta, shift + it)
            #
            mean_lea = 0
            mean_lda = 0
            mean_lec = 0
            mean_ldc = 0
            mean_loss = 0
            mean_found = 0
            mean_correct = 0
            mean_good_pairs = 0
