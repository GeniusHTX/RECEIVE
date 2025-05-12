# coding: utf-8
import warnings
import setproctitle
from Trainer import MothTrainer
import argparse
import numpy as np
import os
import time
import torch
from util import get_norm
from DATASETS import get_dataloader, get_backdoored_loader
warnings.filterwarnings('ignore', category=FutureWarning)
torch.multiprocessing.set_sharing_strategy('file_system')

proc_title = "iterative_trigger_inversion"
setproctitle.setproctitle(proc_title)

def load_data():
    # memory loader is just for test on knn
    _memory_loader = get_dataloader(
        args.dataset, True, 1.0,
        batch_size=args.batch_size
    )
    re_loader = get_dataloader(
        args.dataset, True, args.data_ratio,
        batch_size=args.batch_size
    )
    test_backdoor_loader = get_backdoored_loader(
        args.dataset, train=False, ratio=0.05,
        batch_size=args.batch_size,
        trigger_file=args.trigger_file,
        reference_label=args.reference_label
    )
    test_clean_loader = get_dataloader(args.dataset, False, 0.05)
    return _memory_loader, re_loader, test_clean_loader, test_backdoor_loader



def main():
    if args.phase == 'moth':
        with open(f'{args.log_dir}/setting.txt', 'w+') as f:
            f.write(str(args))
        trainer = MothTrainer(
            args=args,
            trigger_steps=args.re_steps
        )
        trainer.mothHarden()
    else:
        print('Option [{}] is not supported!'.format(args.phase))


# def main_backup():
#     moth()


def parse():
    parser = argparse.ArgumentParser(description='Process input arguments.')
    # shared params
    parser.add_argument('--gpu', default=2, type=int, help='gpu id')
    parser.add_argument('--seed', default=0, help='seed index', type=int)
    parser.add_argument('--phase', default='moth', help='phase of framework')
    parser.add_argument('--dataset', default='cifar10', help='dataset, similar to encoder usage info')
    parser.add_argument('--model', default='resnet18', help='model architecture')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--epochs', default=2, type=int, help='hardening epochs')
    parser.add_argument('--log_dir', default='MSE-test-loss', type=str,
                        help='To storage the mask pattern and loss')
    parser.add_argument('--results_dir', default='ckpts', help='The directory to restore our model')
    parser.add_argument('--knn-t', default=0.5, type=float, help='softmax temperature in kNN monitor')
    parser.add_argument('--knn-k', default=200, type=int, help='k in kNN monitor')

    # moth params
    parser.add_argument('--pretrained_encoder',
                        default='ckpts/reference_be/cifar10/stl10_backdoored_encoder/model_200.pth',
                        help='ckpt path')
    parser.add_argument('--harden_lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--re_lr', default=5e-1, type=float, help='the NC learning rate')
    parser.add_argument('--data_ratio', default=0.01, type=float,
                        help='ratio of training samples for hardening, including inversion and unlearn')
    parser.add_argument('--encoder_usage_info', default='cifar10', type=str,
                        help='The dataset we use to train the pretrained encoder, decide the transform type')
    parser.add_argument('--attack_threshold', type=float, default=0.99, help='')
    parser.add_argument('--similarity_threshold', type=float, default=0.90, help='')
    parser.add_argument('--init_cost', type=float, default=5e-2, help='')
    parser.add_argument('--pair', default='0-0', help='label pair')
    parser.add_argument('--attack_size', default=32, help='used to inversion trigger')
    parser.add_argument('--re_steps', default=300, type=int, help='used to inversion trigger, similar to epoch')
    parser.add_argument('--downstreamTask', default='stl10', help='downstream task')

    # Early stopping
    parser.add_argument('--epsilon', default=0.001, type=float, help='to decide early stop')
    parser.add_argument('--patience', default=7, type=int, help='to decide early stop')

    # attack knowledge
    parser.add_argument('--reference_label', default=9, type=int,
                        help='target class in the target downstream task')
    parser.add_argument('--trigger_file', default='trigger/trigger_pt_white_21_10_ap_replace.npz', type=str,
                        help='path to the trigger file (default: none)')
    parser.add_argument('--reference_file', default='reference/cifar10/truck.npz', type=str,
                        help='path to the reference file, the target class of downstream')
    parser.add_argument('--source_label', default=-1, type=int,
                        help='victim class in the target downstream task')
    # parser.add_argument('--portion', default=1.0, type=float, help='the portion of samples to stamp a trigger')
    # parser.add_argument('--beta', default=1.0, type=float, help='the weight of clean loss and debad loss')
    return parser.parse_args()


if __name__ == '__main__':
    # load params
    args = parse()

    # set gpu usage
    args.n_gpu = torch.cuda.device_count()

    # set random seed
    SEED = [1024, 557540351, 157301989, 100, 1234]
    SEED = SEED[args.seed]
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # set basics
    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    args.mean, args.std = get_norm(args.dataset)

    # main function
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)
    print(args)
    time_start = time.time()
    main()
    # moth()
    time_end = time.time()
    print('=' * 50)
    print('Running time:', (time_end - time_start) / 60, 'm')
    print('=' * 50)
