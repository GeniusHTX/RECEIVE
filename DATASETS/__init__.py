from .dataset import GTSRB, CelebA_attr, BadCIFAR, BadSTL, BadSTL_targeted, BadSVHN, BadGTSRB
from .ContrastiveSet import PAIR
from .utilData import *
import torch
from torchvision import datasets, transforms
from util import _mean, _size, _std


def get_contrastive_data(data, name='cifar10'):
    """
    :param name: the name of dataset
    :param data: Dataset class
    :return: dataloader
    """
    if name == 'cifar10':
        pretrain_data = PAIR(
            dataset=data,
            transform=train_transform_cifar10
        )
    elif name == 'stl10':
        pretrain_data = PAIR(
            dataset=data,
            transform=train_transform_stl10
        )
    else:
        pretrain_data = None
        ValueError(f"{name} is not supported!")
    return pretrain_data


def get_dataset(dataset, train=True, ratio=1.0, encoder_usage_info='cifar10'):
    transforms_list = [
        transforms.Resize(_size[encoder_usage_info][:2]),
        transforms.ToTensor(),
        transforms.Normalize(_mean[encoder_usage_info], _std[encoder_usage_info])
    ]
    transform = transforms.Compose(transforms_list)

    data_root = 'data'
    #  channel first
    if dataset == 'gtsrb':
        dataset = GTSRB(data_root, train, transform)
    elif dataset == 'mnist':
        dataset = datasets.MNIST(data_root, train, transform, download=True)
    elif dataset == 'cifar10':
        # classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        dataset = datasets.CIFAR10(data_root, train, transform, download=True)
    elif dataset == 'celeba':
        dataset = CelebA_attr(data_root, train, transform)
    elif dataset == 'stl10':
        # classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
        di = {True: 'train', False: 'test'}
        dataset = datasets.STL10(data_root, di[train], transform=transform, download=True)
    elif dataset == 'svhn':
        di = {True: 'train', False: 'test'}
        dataset = datasets.SVHN(data_root, di[train], transform, download=True)
    elif dataset == 'gtrsb':
        dataset = GTSRB(data_root, train, transform)
    else:
        raise Exception('Invalid dataset')
    if ratio < 1:
        indices = np.arange(int(len(dataset) * ratio))
        torch.manual_seed(100)
        dataset = torch.utils.data.Subset(dataset, indices)
    return dataset


def get_dataloader(dataset, train=True, ratio=1.0, batch_size=128, encoder_usage_info='cifar10'):
    dataset = get_dataset(
        dataset=dataset,
        train=train,
        ratio=ratio,
        encoder_usage_info=encoder_usage_info
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             num_workers=6, shuffle=train)
    return dataloader


def get_backdoored_loader(
        dataset, trigger_file,
        train=True, ratio=1.0, batch_size=128, reference_label=-1,
        mode='non-targeted',
        encoder_usage_info='cifar10'
):
    # this transforms is used for normalization
    transforms_list = [
        transforms.Resize(_size[encoder_usage_info][:2]),
        transforms.ToTensor(),
        transforms.Normalize(_mean[encoder_usage_info], _std[encoder_usage_info])
    ]
    transform = transforms.Compose(transforms_list)
    if dataset == 'cifar10' and mode == 'non-targeted':
        dataset = BadCIFAR(
            data_root='data',
            train=train,
            transform=transform,
            trigger_file=trigger_file,
            reference_label=reference_label
        )
    elif dataset == 'stl10' and mode == 'non-targeted':
        dataset = BadSTL(
            data_root='data',
            train=train,
            transform=transform,
            trigger_file=trigger_file,
            reference_label=reference_label
        )
    elif dataset == 'stl10' and mode == 'targeted':
        dataset = BadSTL_targeted(
            data_root='data',
            train=train,
            transform=transform,
            trigger_file=trigger_file,
            reference_label=-1
        )
    elif dataset == 'svhn' and mode == 'non-targeted':
        dataset = BadSVHN(
            data_root='data',
            train=train,
            transform=transform,
            trigger_file=trigger_file,
            reference_label=reference_label
        )
    elif dataset == 'gtsrb' and mode == 'non-targeted':
        dataset = BadGTSRB(
            data_root='data',
            train=train,
            transform=transform,
            trigger_file=trigger_file,
            reference_label=reference_label
        )
    else:
        raise Exception(f'Invalid dataset: {dataset}')
    if ratio < 1:
        indices = np.arange(int(len(dataset) * ratio))
        dataset = torch.utils.data.Subset(dataset, indices)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             num_workers=6, shuffle=train)
    return dataloader
