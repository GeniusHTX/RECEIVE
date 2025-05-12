from torch.utils.data import Dataset
from util import deprocess


class PAIR(Dataset):
    def __init__(self, dataset, transform=None):
        """
        :param dataset: Dataset， preprocessed tensor
        :param transform: data argumentation
        """
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, item):
        im, _ = self.dataset[item]
        if item < 10:
            import torchvision
            torchvision.utils.save_image(im, f'Temp_data/ori{item}.png')
            import torch
            im_ = im.unsqueeze(0)
            # print(im_.shape)
            torchvision.utils.save_image(deprocess(self.transform(im_), dataset='stl10'),
                                         f'Temp_data/tran1-{item}.png')
            torchvision.utils.save_image(deprocess(self.transform(im_), dataset='stl10'),
                                         f'Temp_data/tran2-{item}.png')
        # import sys
        # sys.exit(-1)
        return self.transform(im), self.transform(im)

    def __len__(self):
        return len(self.dataset)
