import math
import torch
import numpy as np
from sklearn.metrics import r2_score
from scipy.spatial.transform import Rotation


def compute_rte(t, t_est):

    t = t.squeeze().detach().cpu().numpy()
    t_est = t_est.squeeze().detach().cpu().numpy()

    return np.linalg.norm(t - t_est)


def compute_rre(R_est, R):

    eps=1e-16

    R = R.squeeze().detach().cpu().numpy()
    R_est = R_est.squeeze().detach().cpu().numpy()

    return np.arccos(
        np.clip(
            (np.trace(R_est.T @ R) - 1) / 2,
            -1 + eps,
            1 - eps
        )
    ) * 180. / math.pi


# Metrics
def npmat2euler(mats, seq='zyx'):
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_dcm(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')



def compute_metrics(true_r, pred_r, true_t, pred_t):
    pred_euler = npmat2euler(pred_r)
    true_euler = npmat2euler(true_r)

    r_mse = np.mean((pred_euler - true_euler) ** 2)
    r_rmse = np.sqrt(r_mse)

    r_mae = np.mean(np.abs(pred_euler - true_euler))

    t_mse = np.mean((pred_t - true_t) ** 2)
    t_rmse = np.sqrt(t_mse)

    t_mae = np.mean(np.abs(pred_t - true_t))

    r_r2_score = r2_score(true_euler, pred_euler)
    t_r2_score = r2_score(true_t, pred_t)

    return {
        'r_mse': r_mse,
        't_mse': t_mse,

        'r_rmse': r_rmse,
        't_rmse': t_rmse,

        'r_mae': r_mae,
        't_mae': t_mae,

        'r_r2_score': r_r2_score,
        't_r2_score': t_r2_score
    }
