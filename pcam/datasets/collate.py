# Part of the code in this file is taken from https://github.com/chrischoy/DeepGlobalRegistration/blob/46dd264580b4191accedc277f4ae434acdb4d380/dataloader/base_loader.py#L24

from pcam.tool.transforms import decompose_rotation_translation
import lightconvpoint.nn as lcp_nn
import torch
import numpy as np

class CollateFunc:
  def __init__(self):
    self.collation_fn = self.collate_pair_fn

  def __call__(self, list_data):
    return self.collation_fn(list_data)

  def collate_pair_fn(self, list_data):
    N = len(list_data)

    list_data = [data for data in list_data if data is not None]
    if N != len(list_data):
      logging.info(f"Retain {len(list_data)} from {N} data.")
    if len(list_data) == 0:
      raise ValueError('No data in the batch')

    xyz0_full, xyz1_full, xyz0, xyz1, trans, inv_trans, one_one_attention, filename, p0_mean, p1_mean = list(
      zip(*list_data)
    )

    trans_batch = torch.from_numpy(np.stack(trans)).float()
    inv_trans_batch = torch.from_numpy(np.stack(inv_trans)).float()

    pts1 = torch.from_numpy(np.stack(xyz0)).float()
    pts2 = torch.from_numpy(np.stack(xyz1)).float()
    raw_pts1 = torch.from_numpy(np.stack(xyz0_full)).float()
    raw_pts2 = torch.from_numpy(np.stack(xyz1_full)).float()

    p0_mean = torch.from_numpy(np.stack(p0_mean)).float().transpose(1, 2)
    p1_mean = torch.from_numpy(np.stack(p1_mean)).float().transpose(1, 2)

    Rs, ts = decompose_rotation_translation(trans_batch)
    Rs_inv, ts_inv = decompose_rotation_translation(inv_trans_batch)

    ts = ts.unsqueeze(-1)
    ts_inv = ts_inv.unsqueeze(-1)

    search = lcp_nn.SearchQuantized(K=32, stride=1)
    pts1 = pts1.transpose(1, 2)
    pts2 = pts2.transpose(1, 2)
    indices1, pts1 = search(pts1)
    indices2, pts2 = search(pts2)

    one_one_attention = torch.from_numpy(np.stack(one_one_attention)).float()

    return {
      "raw_src": raw_pts1,
      "raw_tgt": raw_pts2,
      "src": pts1,
      "tgt": pts2,
      "filename": filename,
      "attention": one_one_attention,
      "indices1": indices1,
      "indices2": indices2,
      "rotation": Rs,
      "translation": ts,
      "inv_rotation": Rs_inv,
      "inv_translation": ts_inv,
      "p0_mean": p0_mean,
      "p1_mean": p1_mean,
    }
