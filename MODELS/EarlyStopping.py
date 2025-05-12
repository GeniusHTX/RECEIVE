import numpy as np


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, log_dir, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation distance improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation distance improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.log_dir = log_dir
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = 0
        self.best_mask_pattern_idx = 0
        self.early_stop = False
        self.metric_min = 0.0
        self.delta = delta

    def __call__(self, metric, cur_idx, mask=None, pattern=None):
        """ metric is like a score, equal to distance """
        score = metric
        pre_best = 0
        if score > self.best_score:
            pre_best = self.best_score
            self.best_score = score
            self.best_mask_pattern_idx = cur_idx
            self.save_mask_pattern(metric, cur_idx, mask, pattern)
            self.counter = 0 if abs(pre_best - self.best_score) > self.delta else 1
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}, idx-{cur_idx}')

            if self.counter >= self.patience:
                self.early_stop = True

    def save_mask_pattern(self, metric, cur_idx, mask, pattern):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'distance improved ({self.metric_min:.3f} --> {metric:.3f}).'
                  f'  Saving the reversed mask and pattern ...')
        np.savez(f'{self.log_dir}/mask-pattern-best.npz',
                 mask=mask, pattern=pattern, idx=cur_idx)
        self.metric_min = metric
