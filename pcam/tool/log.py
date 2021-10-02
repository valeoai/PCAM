import numpy as np
from collections import defaultdict
from .loss import compute_rre, compute_rte


class Logger:

    def __init__(self):
        self.store = defaultdict(list)

    def reset(self):
        self.store = defaultdict(list)

    def add(self, key, value):
        self.store[key].append(value)
    
    def avg(self, key):
        return np.mean(self.store[key])

    def save_sacred(self, ex, it):
        for k, v in self.store.items():
            if "fail" in k:
                ex.log_scalar(k, np.sum(v), it)
            else:
                ex.log_scalar(k, np.mean(v), it)

    def show(self):

        print("\n========================")

        for k, v in self.store.items():
            
            if "fail" in k:
                print(k, np.sum(v))
            else:
                print(k, np.mean(v))
                
            if "attn_acc" in k or "count" in k:
                print("min " + k + ": ", np.min(v))
                print("max " + k + ": ", np.max(v))
            
        print("========================\n")
            

def save_metrics(logger, prefix, R, t, R_est, t_est, te_thres=0.6, re_thres=5):

    bs = R.shape[0]
    rot_error = compute_rre(R_est, R)
    trans_error = compute_rte(t.reshape(bs, -1), t_est.reshape(bs, -1))
    logger.add(prefix + ".rte_all", trans_error)
    logger.add(prefix + ".rre_all", rot_error)  

    if rot_error < re_thres and trans_error < te_thres:
        logger.add(prefix + ".recall", 1)
        logger.add(prefix + ".rte", trans_error)
        logger.add(prefix + ".rre", rot_error)            
    else:
        if rot_error > re_thres and trans_error > te_thres:
            logger.add(prefix + ".fail_both", 1)
        elif rot_error > 5:
            logger.add(prefix + ".fail_rot", 1)
        else:
            logger.add(prefix + ".fail_trans", 1)
        logger.add(prefix + ".recall", 0)

