# coding: utf-8
import torch
from typing import Tuple
from torch.utils.data import Dataset
from torchvision import transforms

_mean = {
    'default': [0.5, 0.5, 0.5],
    'mnist': [0.5, 0.5, 0.5],
    'cifar10': [0.4914, 0.4822, 0.4465],
    'gtsrb': [0.0, 0.0, 0.0],
    'celeba': [0.0, 0.0, 0.0],
    'imagenet': [0.485, 0.456, 0.406],
    'stl10': [0.44087798, 0.42790666, 0.38678814],
    'svhn': [0.4376821046090723, 0.4437697045639686, 0.4728044222297267],
}

_std = {
    'default': [0.5, 0.5, 0.5],
    'mnist': [0.5, 0.5, 0.5],
    'cifar10': [0.2471, 0.2435, 0.2616],
    'gtsrb': [1.0, 1.0, 1.0],
    'celeba': [1.0, 1.0, 1.0],
    'imagenet': [0.229, 0.224, 0.225],
    'stl10': [0.25507198, 0.24801506, 0.25641308],
    'svhn': [0.19803012447157134, 0.20101562471828877, 0.19703614172172396],
}

_size = {
    'mnist': (28, 28, 1),
    'cifar10': (32, 32, 3),
    'gtsrb': (32, 32, 3),
    'celeba': (64, 64, 3),
    'imagenet': (224, 224, 3),
    'stl10': (32, 32, 3),  
    'svhn': (32, 32, 3),
}

_num = {
    'mnist': 10,
    'cifar10': 10,
    'gtsrb': 43,
    'celeba': 8,
    'imagenet': 1000,
    'stl10': 10,
    'svhn': 10,
}

argumentation = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(p=0.5),
    # transforms.GaussianBlur(5, sigma=(0.1, 0.5)),
    # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomGrayscale(p=0.2),
    # transforms.PILToTensor()
])


def get_norm(dataset):
    mean, std = _mean[dataset], _std[dataset]
    mean_t = torch.Tensor(mean)
    std_t = torch.Tensor(std)
    return mean_t, std_t


def preprocess(x, dataset, clone=True, channel_first=True):
    if torch.is_tensor(x):
        x_out = torch.clone(x) if clone else x
    else:
        x_out = torch.FloatTensor(x)

    if x_out.max() > 100:
        x_out = x_out / 255.

    if channel_first:
        x_out = x_out.permute(0, 2, 3, 1)

    mean_t, std_t = get_norm(dataset)
    mean_t = mean_t.to(x_out.device)
    std_t = std_t.to(x_out.device)

    x_out = (x_out - mean_t) / std_t
    x_out = x_out.permute(0, 3, 1, 2)
    return x_out


def deprocess(x, dataset, clone=True):
    mean_t, std_t = get_norm(dataset)
    mean_t = mean_t.to(x.device)
    std_t = std_t.to(x.device)
    x_out = torch.clone(x) if clone else x
    x_out = x_out.permute(0, 2, 3, 1) * std_t + mean_t
    x_out = x_out.permute(0, 3, 1, 2)
    return x_out


def get_num(dataset):
    return _num[dataset]


def get_size(dataset):
    return _size[dataset]


def pgd_attack(model, images, labels, mean, std,
               eps=0.3, alpha=2 / 255, iters=40):
    loss = torch.nn.CrossEntropyLoss()

    ori_images = images.data

    images = images + 2 * (torch.rand_like(images) - 0.5) * eps
    images = torch.clamp(images, 0, 1)

    mean = mean.to(images.device)
    std = std.to(images.device)

    for i in range(iters):
        images.requires_grad = True

        outputs = model(
            ((images.permute(0, 2, 3, 1) - mean) / std) \
                .permute(0, 3, 1, 2)
        )

        model.zero_grad()
        cost = loss(outputs, labels)
        cost.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return images


def loader_to_tensor(loader) -> Tuple[torch.Tensor, torch.Tensor]:
    x_extra = y_extra = None
    for idx, (x_batch, y_batch) in enumerate(loader):
        if idx == 0:
            x_extra = x_batch
            y_extra = y_batch
        else:
            x_extra = torch.cat([x_extra, x_batch])
            y_extra = torch.cat([y_extra, y_batch])
    return x_extra, y_extra


class PackageDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x_tensor = x_tensor
        self.y_tensor = y_tensor
        # print(type(y_tensor))

    def __getitem__(self, item):
        return self.x_tensor[item], self.y_tensor[item]

    def __len__(self):
        return self.x_tensor.size(0)
