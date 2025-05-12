import os
import argparse
import random
import torchvision
import numpy as np
from functools import partial
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import OrderedDict
from tqdm import tqdm
import torch
import torch.nn as nn
from util import loader_to_tensor, PackageDataset
import torch.nn.functional as F
from MODELS import get_encoder_architecture_usage
from util import get_num
from DATASETS import get_dataloader, get_backdoored_loader
from EVALUATION import create_torch_dataloader, NeuralNet, net_train, net_test, predict_feature
import setproctitle
torch.multiprocessing.set_sharing_strategy('file_system')

proc_title = "train_downstream_clf"
setproctitle.setproctitle(proc_title)


def parse():
    parser = argparse.ArgumentParser(description='Evaluate the clean or backdoored encoders')
    parser.add_argument('--dataset', default='cifar10', type=str, help='downstream dataset')
    parser.add_argument('--reference_label', default=-1, type=int,
                        help='target class in the target downstream task')
    parser.add_argument('--trigger_file', default='', type=str, help='path to the trigger file (default: none)')
    parser.add_argument('--encoder_usage_info', default='', type=str,
                        help='used to locate encoder usage info, e.g., encoder architecture and input normalization '
                             'parameter')
    parser.add_argument('--encoder', default='', type=str, help='path to the image encoder')

    parser.add_argument('--gpu', default=0, type=int, help='the index of gpu used to train the model')
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--seed', default=100, type=int, help='seed')
    parser.add_argument('--nn_epochs', default=500, type=int)
    parser.add_argument('--hidden_size_1', default=512, type=int)
    parser.add_argument('--hidden_size_2', default=256, type=int)
    parser.add_argument('--results_dir', default='reproduction/cifar10/downstreamclf', type=str)
    parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='mini-batch size')
    # no use any more
    parser.add_argument('--source_label', default=-1, type=int,
                        help='victim class in the target downstream task')
    parser.add_argument('--mode', default='non-targeted', help='targeted attack or non-targeted attack')
    return parser.parse_args()


def set_settings():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


def load_data():
    print('reference label:', args.reference_label)
    _test_loader_clean = get_dataloader(
        args.dataset, train=False, ratio=1.0, batch_size=args.batch_size,
        encoder_usage_info=args.encoder_usage_info
    )
    _test_loader_backdoor = get_backdoored_loader(
        args.dataset, train=True, ratio=0.5,
        batch_size=args.batch_size,
        trigger_file=args.trigger_file,
        reference_label=args.reference_label,
        mode=args.mode,
        encoder_usage_info=args.encoder_usage_info
    )
    _train_loader = get_dataloader(
        args.dataset, train=True, batch_size=args.batch_size,
        encoder_usage_info=args.encoder_usage_info
    )
    return _train_loader, _test_loader_backdoor, _test_loader_clean



if __name__ == '__main__':
    args = parse()
    set_settings()
    # assert args.reference_label >= 0, 'Enter the correct target class'

    args.data_dir = f'./data'
    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader_backdoor, test_loader_clean = load_data()

    # select backdoor
    # select backdoor loader
    if args.source_label == -1:
        victim_loader = test_loader_backdoor
    else:
        # select victim class
        train_set_x, train_set_y = loader_to_tensor(test_loader_clean)
        victim_indices = (train_set_y == args.source_label)
        train_set_x = train_set_x[victim_indices]
        train_set_y = train_set_y[victim_indices]

        # package dataset
        victim_set = PackageDataset(x_tensor=train_set_x, y_tensor=train_set_y)
        print(len(victim_set))
        # import sys
        # sys.exit(-1)
        from torch.utils.data import DataLoader

        victim_loader = DataLoader(victim_set, batch_size=args.batch_size)
    test_loader_backdoor = victim_loader

    model = get_encoder_architecture_usage(args)
    args.n_gpu = torch.cuda.device_count()
    model = model.cuda(device=args.gpu)
    if args.encoder != '':
        print('Loaded from: {}'.format(args.encoder))
        checkpoint = torch.load(args.encoder, map_location=args.device)['state_dict']
        # checkpoint = torch.load(args.encoder)
        checkpoint = checkpoint.module if hasattr(checkpoint, 'module') else checkpoint
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k.replace('module.', '')  # remove `module.`
            new_state_dict[name] = v
        if args.encoder_usage_info in ['CLIP', 'imagenet'] and 'clean' in args.encoder:
            model.visual.load_state_dict(new_state_dict)
        else:
            if "f.f.0.weight" in new_state_dict.keys():
                model.load_state_dict(new_state_dict)
            else:
                model.f.load_state_dict(new_state_dict)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    if args.encoder_usage_info in ['CLIP', 'imagenet']:
        feature_bank_training, label_bank_training = predict_feature(model.visual, train_loader, args.device)
        feature_bank_testing, label_bank_testing = predict_feature(model.visual, test_loader_clean, args.device)
        feature_bank_backdoor, label_bank_backdoor = predict_feature(model.visual, test_loader_backdoor, args.device)
        # feature_bank_target, label_bank_target = predict_feature(model.visual, target_loader)
    else:
        feature_bank_training, label_bank_training = predict_feature(model.f, train_loader, args.device)
        feature_bank_testing, label_bank_testing = predict_feature(model.f, test_loader_clean, args.device)
        feature_bank_backdoor, label_bank_backdoor = predict_feature(model.f, test_loader_backdoor, args.device)
        # feature_bank_target, label_bank_target = predict_feature(model.f, target_loader)

    nn_train_loader = create_torch_dataloader(feature_bank_training, label_bank_training, args.batch_size)
    nn_test_loader = create_torch_dataloader(feature_bank_testing, label_bank_testing, args.batch_size)
    nn_backdoor_loader = create_torch_dataloader(feature_bank_backdoor, label_bank_backdoor, args.batch_size)

    input_size = feature_bank_training.shape[1]

    criterion = nn.CrossEntropyLoss()
    num_of_classes = get_num(args.dataset)
    net = NeuralNet(input_size, [args.hidden_size_1, args.hidden_size_2], num_of_classes).cuda(device=args.device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    print(args)
    print(num_of_classes, input_size, args.device)
    for epoch in range(1, args.nn_epochs + 1):
        net_train(net, nn_train_loader, optimizer, epoch, criterion, device=args.device)
        if 'clean' in args.encoder:
            net_test(net, nn_test_loader, epoch, criterion, args.device, 'Clean Accuracy (CA)')
            net_test(net, nn_backdoor_loader, epoch, criterion, args.device, 'Attack Success Rate-Baseline (ASR-B)')
        else:
            net_test(net, nn_test_loader, epoch, criterion, args.device, 'Backdoored Accuracy (BA)')
            net_test(net, nn_backdoor_loader, epoch, criterion, args.device, 'Attack Success Rate (ASR)')
    print(f'epoch is {args.nn_epochs}')
    torch.save({'epoch': args.nn_epochs,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict()},
               args.results_dir + '/clf_' + str(args.nn_epochs) + '.pth')

