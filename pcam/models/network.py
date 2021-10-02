import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import lightconvpoint.nn as lcp_nn


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, ConvNet, Search, K, act):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act()

        self.enc1 = lcp_nn.Conv(
            ConvNet(out_channels, out_channels=out_channels, kernel_size=K),
            Search(K=K, stride=1),
            activation=act(),
            normalization=nn.InstanceNorm1d(out_channels, affine=True)
        )

        self.enc2 = lcp_nn.Conv(
            ConvNet(out_channels, out_channels=out_channels, kernel_size=K),
            Search(K=K, stride=1),
            activation=None,
            normalization=nn.InstanceNorm1d(out_channels, affine=True)
        )

        if in_channels != out_channels:
            self.conv = nn.Conv1d(in_channels, out_channels, 1)
            self.norm = nn.InstanceNorm1d(out_channels, affine=True)
        else:
            self.conv = None
            self.norm = None

    def forward(self, input, points, indices=None):

        # Shape
        bs, _, n_points = points.shape

        # Skip part
        if self.conv is not None:
            input = self.norm(self.conv(input))

        # Init.
        x = input
        pts = points

        # Residual part
        x, pts, idx = self.enc1(input=x, points=pts, support_points=pts, indices=indices)
        if x is None:
            return None, pts, idx
        x, pts, idx = self.enc2(input=x, points=pts, support_points=pts, indices=indices)

        # Addition
        x = self.act(input + x)

        return x, pts, idx


class Encoder(nn.Module):

    def __init__(self, ConvNet, Search, n_channels, K):

        super().__init__()

        self.layers = nn.ModuleList(self.create_layers(n_channels, ConvNet, Search, K))

    def create_layers(self, n_channels, ConvNet, Search, K):

        layers = []
        for i in range(len(n_channels) - 1):
            encoder = Block(n_channels[i], n_channels[i + 1], ConvNet, Search, K=K, act=nn.ReLU)
            layers.append(encoder)

        return layers

    def forward(self, x, pts, indices=None):

        for layer in self.layers:
            x, pts, idx = layer(input=x, points=pts, indices=indices)

        return x, pts, idx


class PointMatcher(nn.Module):

    def __init__(self, nb_encoders, last_attention, sparse_attention):

        super().__init__()

        K = 16
        self.last_attention = last_attention
        self.sparse_attention = sparse_attention

        self.encoders = nn.ModuleList()
        if nb_encoders >= 2:
            n_channels = [3, 32, 32, 32]
            self.encoders.append(Encoder(lcp_nn.FKAConv, lcp_nn.SearchQuantized, n_channels, K))
            n_channels = [64, 32, 32, 32]
            self.encoders.append(Encoder(lcp_nn.FKAConv, lcp_nn.SearchQuantized, n_channels, K))
        if nb_encoders >= 4:
            n_channels = [64, 64, 64, 64]
            self.encoders.append(Encoder(lcp_nn.FKAConv, lcp_nn.SearchQuantized, n_channels, K))
            n_channels = [128, 64, 64, 64]
            self.encoders.append(Encoder(lcp_nn.FKAConv, lcp_nn.SearchQuantized, n_channels, K))
        if nb_encoders >= 6:
            n_channels = [128, 128, 128, 128]
            self.encoders.append(Encoder(lcp_nn.FKAConv, lcp_nn.SearchQuantized, n_channels, K))
            n_channels = [256, 128, 128, 128]
            self.encoders.append(Encoder(lcp_nn.FKAConv, lcp_nn.SearchQuantized, n_channels, K))
        if nb_encoders >= 8:
            n_channels = [256, 128, 128, 128]
            self.encoders.append(Encoder(lcp_nn.FKAConv, lcp_nn.SearchQuantized, n_channels, K))
            n_channels = [256, 128, 128, 128]
            self.encoders.append(Encoder(lcp_nn.FKAConv, lcp_nn.SearchQuantized, n_channels, K))
        assert len(self.encoders) == nb_encoders
        print('Nb encoders', len(self.encoders))

    def get_log_attention(self, x1, x2, temp=0.03):

        # Normalize features: B x C x N
        x1_norm = F.normalize(x1, dim=1, p=2)
        x2_norm = F.normalize(x2, dim=1, p=2)

        # Scaled dot product: B x N x N
        scores = torch.bmm(x1_norm.transpose(1, 2), x2_norm) / temp

        # Attention (both direction)
        log_att1 = F.log_softmax(scores, dim=2)
        log_att2 = F.log_softmax(scores, dim=1)

        return log_att1, log_att2

    def encode(self, x1, x2, pts1, pts2, encoder, indices1=None, indices2=None):

        # Extract Point Feature: B x C x N
        x1, pts1, _ = encoder(x1, pts1, indices1)
        x2, pts2, _ = encoder(x2, pts2, indices2)

        # Attentions
        log_att1, log_att2 = self.get_log_attention(x1, x2)

        # New features
        # Side 1
        att1 = torch.exp(log_att1)
        x1_comb = torch.bmm(att1, x2.transpose(1, 2)).transpose(1, 2)
        # Side 2
        att2 = torch.exp(log_att2).transpose(1, 2)
        x2_comb = torch.bmm(att2, x1.transpose(1, 2)).transpose(1, 2)

        # Concat
        x1 = torch.cat([x1, x1_comb], dim=1)
        x2 = torch.cat([x2, x2_comb], dim=1)

        return x1, x2, (log_att1, log_att2)

    def sparsify_attention(self, log_attention, dim):

        # If soft-attention
        if not self.sparse_attention:
            return F.normalize(torch.exp(log_attention), dim=dim, p=1)

        # Find best match
        idx = torch.argmax(log_attention, dim=dim, keepdim=True)

        # Sparse attention
        sparse_attention = torch.zeros_like(log_attention, requires_grad=False)
        sparse_attention.scatter_(dim, idx, 1.)

        return sparse_attention

    def forward(self, pts1, pts2, indices1=None, indices2=None):

        # Compute series of attention matrices
        log_attns = []
        x1, x2 = pts1, pts2
        for encoder in self.encoders:
            x1, x2, log_att = self.encode(
                x1, x2, pts1, pts2, encoder, indices1=indices1, indices2=indices2
            )
            log_attns.extend(log_att)

        # Sum ***log*** of attentions
        if self.last_attention:
            log_attn_row = log_attns[-2]  # Row normalised attention are in even position
            if self.training:
                log_attn_col = log_attns[-1]  # Column normalised attention are in odd position
            else:
                log_attn_col = None
        else:
            log_attn_row = sum(log_attns[::2])  # Row normalised attention are in even position
            if self.training:
                log_attn_col = sum(log_attns[1::2])  # Column normalised attention are in odd position
            else:
                log_attn_col = None

        # Selection of best match
        attention_row = self.sparsify_attention(log_attn_row, 2)
        if self.training:
            attention_col = self.sparsify_attention(log_attn_col, 1).transpose(1, 2)

        # Corresponding points
        corres_pts_for_pts1 = torch.bmm(attention_row, pts2.transpose(1, 2)).transpose(1, 2)
        if self.training:
            corres_pts_for_pts2 = torch.bmm(attention_col, pts1.transpose(1, 2)).transpose(1, 2)
        else:
            corres_pts_for_pts2 = None

            # Output
        out = [corres_pts_for_pts1, corres_pts_for_pts2, log_attn_row, log_attn_col]

        return out


class ConfidenceEstimator(nn.Module):

    def __init__(self):

        super().__init__()

        # Param.
        K = 16
        n_channels1 = [6, 16, 16, 16, 32, 32, 32, 64, 64, 64]

        # Layers
        self.encoder1 = Encoder(lcp_nn.FKAConv, lcp_nn.SearchQuantized, n_channels1, K)
        self.cv4 = lcp_nn.Conv(
            lcp_nn.FKAConv(n_channels1[-1], out_channels=1, kernel_size=K),
            lcp_nn.SearchQuantized(K=K, stride=1),
            activation=nn.Sigmoid(),
            normalization=None
        )

    def forward(self, in1, in2, pts, indices=None):

        # Concat input
        x = torch.cat([in1, in2], dim=1)

        # Network
        x, pts, _ = self.encoder1(x, pts, indices)
        x, pts, indices = self.cv4(input=x, points=pts, support_points=pts, indices=indices)

        return x


class PoseEstimator(nn.Module):

    def __init__(self, nb_encoders, last_attention, sparse_attention, backprop, threshold=None, N=None):

        super().__init__()

        #
        self.backprop = backprop

        # Point Matcher
        self.matcher = PointMatcher(nb_encoders, last_attention, sparse_attention)

        # Confidence estimator for input pairs
        self.confid = ConfidenceEstimator()

        # Threshold
        self.threshold = threshold
        print('Threshold', self.threshold)
        self.N = N
        print('N', self.N)

    @torch.no_grad()
    def estimate_rot_trans(self, x, y, w):

        # L1 normalisation
        if self.N is not None:
           val, ind = torch.topk(w, self.N, dim=-1)
           w = torch.zeros_like(w)
           w.scatter_(-1, ind, val)
        if self.threshold is not None:
           w = w * (w > self.threshold).float()
        w = F.normalize(w, dim=-1, p=1)

        # Center point clouds
        mean_x = (w * x).sum(dim=-1, keepdim=True)
        mean_y = (w * y).sum(dim=-1, keepdim=True)
        x_centered = x - mean_x
        y_centered = y - mean_y

        # Covariance
        cov = torch.bmm(y_centered, (w * x_centered).transpose(1, 2))

        # Rotation
        U, _, V = torch.svd(cov)
        #det = torch.det(U) * torch.det(V)
        det = det_3x3(U) * det_3x3(V)
        S = torch.eye(3, device=U.device).unsqueeze(0).repeat(x.shape[0], 1, 1)
        S[:, -1, -1] = det
        R = torch.bmm(U, torch.bmm(S, V.transpose(1, 2)))

        # Translation
        T = mean_y - torch.bmm(R, mean_x)

        return R, T, w

    def forward(self, pts1, pts2, indices1=None, indices2=None):

        # Match points
        out = self.matcher(pts1, pts2, indices1=indices1, indices2=indices2)
        corres_pts_for_pts1, corres_pts_for_pts2, log_attn_row, log_attn_col = out

        # Confidence estimator
        if self.backprop:
            w_pts1 = self.confid(pts1, corres_pts_for_pts1, pts1, indices1)
            if self.training:
                w_pts2 = self.confid(pts2, corres_pts_for_pts2, pts2, indices2)
            else:
                w_pts2 = None
        else:
            w_pts1 = self.confid(pts1, corres_pts_for_pts1.detach(), pts1, indices1)
            if self.training:
                w_pts2 = self.confid(pts2, corres_pts_for_pts2.detach(), pts2, indices2)
            else:
                w_pts2 = None

        # Rotation / Translation
        if self.training:
            R, T = None, None  # R and T not involved at training time
        else:
            # Estimate transformation
            R, T, w_pts1 = self.estimate_rot_trans(pts1, corres_pts_for_pts1, w_pts1)

        return R, T, corres_pts_for_pts1, corres_pts_for_pts2, log_attn_row, log_attn_col, w_pts1, w_pts2


def det_3x3(mat):

    a, b, c = mat[:, 0, 0], mat[:, 0, 1], mat[:, 0, 2]
    d, e, f = mat[:, 1, 0], mat[:, 1, 1], mat[:, 1, 2]
    g, h, i = mat[:, 2, 0], mat[:, 2, 1], mat[:, 2, 2]

    det = a * e * i + b * f * g + c * d * h
    det = det - c * e * g - b * d * i - a * f * h

    return det
