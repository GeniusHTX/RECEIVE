from .moth import MothTrainer
import torch
from EVALUATION import knn_test as test
import numpy as np


class NCTrainer(MothTrainer):
    def __init__(self, args):
        super(NCTrainer, self).__init__(args, trigger_steps=args.re_steps)

    def Reverse(self):
        self.model.eval()
        print(self.args.device)
        print('load downstream task for KNN monitor: ', self.args.downstreamTask)
        _memory_loader, re_loader, test_clean_loader, test_backdoor_loader = self.load_data()
        test(
            net=self.model.f,
            memory_data_loader=_memory_loader,
            test_data_backdoor_loader=test_backdoor_loader,
            test_data_clean_loader=test_clean_loader,
            epoch=0,
            args=self.args
        )
        x_extra = y_extra = None
        for idx, (x_batch, y_batch) in enumerate(re_loader):
            if idx == 0:
                x_extra, y_extra = x_batch, y_batch
            else:
                x_extra = torch.cat((x_extra, x_batch))
                y_extra = torch.cat((y_extra, y_batch))
        print(f're data size:{x_extra.shape[0]}')
        source = target = None
        mask, pattern, reference_distance, reg_best, _ = self.trigger.generate(
            (source, target), x_extra, y_extra, steps=self.trigger_steps,
            round_idx=0, draw=False,
        )
        saved_mask, saved_pattern = mask.detach().cpu().numpy(), pattern.detach().cpu().numpy()
        # print(saved_mask.shape, saved_pattern.shape) 3 x 32 x 32
        np.savez(f'{self.args.log_dir}/mask-pattern-NC.npz',
                 mask=saved_mask, pattern=saved_pattern)
        return saved_mask, saved_pattern

