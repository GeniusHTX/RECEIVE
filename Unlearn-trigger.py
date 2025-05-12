# coding: utf-8

import warnings
import setproctitle
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import os
import sys
import time
import torch
import argparse
from util import (
    get_norm,
    preprocess,
    deprocess,
    get_size
)
import torch.nn as nn
from DATASETS import get_dataloader, get_backdoored_loader
from EVALUATION import knn_test
from MODELS import get_encoder_architecture_usage
from torch.autograd import Function
from scipy.stats import wasserstein_distance


def fix_bn(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        m.eval()


proc_title = "Unlearn"
setproctitle.setproctitle(proc_title)
warnings.filterwarnings('ignore', category=FutureWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

argumentation = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomGrayscale(p=0.2),
    # transforms.PILToTensor()
])





def new_preprocess(x, dataset, clone=True, channel_first=True):
    x = preprocess(x, dataset, clone=clone, channel_first=channel_first)
    # x = argumentation(x)
    return x


def loss_similarity(feature1, feature2):
    feature1 = F.normalize(feature1, dim=-1)
    feature2 = F.normalize(feature2, dim=-1)
    # sim = -  torch.exp(torch.sum(feature1 * feature2, dim=-1).mean() / args.knn_t)
    sim = - torch.sum(feature1 * feature2, dim=-1).mean()
    return sim


def contrastive_loss(feature1, feature2):
    cur_batch_size = feature1.size(0)
    out_1 = F.normalize(feature1, dim=-1)
    out_2 = F.normalize(feature2, dim=-1)
    # [2*B, D]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / args.knn_t)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * cur_batch_size, device=sim_matrix.device)).bool()
    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * cur_batch_size, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / args.knn_t)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss


def test_init_sim(mask_re, pattern_re, model, train_loader):
    from util import loader_to_tensor
    model.train()
    # model.apply(fix_bn)
    # model.eval()
    with torch.no_grad():
        x_raw, _ = loader_to_tensor(train_loader)
        x_raw = deprocess(x_raw, args.encoder_usage_info).to(args.device)
        x_raw = x_raw.to(args.device)
        x_re = (1 - mask_re) * x_raw + mask_re * pattern_re
        x_re = torch.clip(x_re, 0.0, 1.0)
        x_re = preprocess(x_re, args.encoder_usage_info)
        feature_raw = model.f(x_raw)
        feature_re = model.f(x_re)
        init_sim = - loss_similarity(feature_re, feature_raw).item()
        print('init sim: {:.3f}'.format(init_sim))
        x_raw_1 = argumentation(x_raw)
        x_raw_2 = argumentation(x_raw)
        x_raw_1 = preprocess(x_raw_1, args.encoder_usage_info)
        x_raw_2 = preprocess(x_raw_2, args.encoder_usage_info)
        feature_1 = model.f(x_raw_1)
        feature_2 = model.f(x_raw_2)
        test_sim = - loss_similarity(feature_1, feature_2).item()
        print('test sim: {:.8f}'.format(test_sim))


def load_data():
    train_loader = get_dataloader(
        args.dataset, True, args.data_ratio, encoder_usage_info=args.encoder_usage_info,
        batch_size=args.batch_size
    )
    # test loader for test downstream task ASR
    test_backdoor_loader = get_backdoored_loader(
        args.downstreamTask, train=False, ratio=1.0,
        batch_size=512,
        trigger_file=args.trigger_file,
        reference_label=args.reference_label,
        encoder_usage_info=args.encoder_usage_info
    )
    # test loader for test downstream task Acc
    test_clean_loader = get_dataloader(args.downstreamTask, False, 1.0, batch_size=512)
    # load memory loader for test use
    _memory_loader = get_dataloader(
        args.downstreamTask, True, 1.0,
        batch_size=512,
        encoder_usage_info=args.encoder_usage_info
    )
    return train_loader, test_backdoor_loader, test_clean_loader, _memory_loader


def moth():
    print(args)
    # num_classes = get_num(args.dataset)
    img_rows, img_cols, img_channels = get_size(args.dataset)

    # load reversed trigger
    arr = np.load(args.re_mask_pattern)
    mask_re, pattern_re = arr['mask'], arr['pattern']  # 3 x 32 x 32
    mask_re, pattern_re = torch.from_numpy(mask_re).cuda(args.device), torch.from_numpy(pattern_re).cuda(args.device)
    print('Reversed trigger size: {:.2f}'.format(torch.sum(torch.abs(mask_re)) / img_channels))
    arr = np.load(r'trigger/trigger_pt_white_21_10_ap_replace.npz')
    mask_gt, pattern_gt = arr['tm'][0].transpose(2, 0, 1), arr['t'][0].transpose(2, 0, 1) / 255
    mask_gt = torch.from_numpy(mask_gt).cuda(args.device).float()
    pattern_gt = torch.from_numpy(pattern_gt).cuda(args.device).float()
    print("mask-pattern-loaded")

    # load model
    model = get_encoder_architecture_usage(args)
    if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10':
        checkpoint = torch.load(args.pretrained_encoder)
        if "f.f.0.weight" in checkpoint['state_dict'].keys():
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.f.load_state_dict(checkpoint['state_dict'])
    elif args.encoder_usage_info == 'imagenet' or args.encoder_usage_info == 'CLIP':
        checkpoint = torch.load(args.pretrained_encoder)
        model.visual.load_state_dict(checkpoint['state_dict'])
    else:
        raise NotImplementedError()
    model = model.to(args.device)

    # load data
    # train loader for unlearn
    train_loader, test_backdoor_loader, test_clean_loader, _memory_loader = load_data()

    # preparation
    test_init_sim(mask_re, pattern_re, model, train_loader)
    # import sys
    # sys.exit(-1)

    # start hardening
    print('=' * 80)
    print('start Unlearning...')
    loss_list = []
    # time_start = time.time()
    # criterion = nn.MSELoss()

    # set loss function and optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                nesterov=True)
    criterion_debackdoor = loss_similarity
    criterion_clean = contrastive_loss

    # test init acc and asr
    model.eval()
    knn_test(
        net=model.f,
        memory_data_loader=_memory_loader,
        test_data_backdoor_loader=test_backdoor_loader,
        test_data_clean_loader=test_clean_loader,
        epoch=0,
        args=args
    )

    seed = 0
    for epoch in range(args.epochs):
        total_loss, total_loss_debad, total_loss_clean = 0.0, 0.0, 0.0
        total_num = 0
        train_bar = tqdm(train_loader)
        model.train()
        # model.apply(fix_bn)
        # print(len(train_loader))
        for (x_batch, _) in train_bar:
            x_batch = x_batch.to(args.device)
            # deprocessed
            x_batch = deprocess(x_batch, clone=False, dataset=args.encoder_usage_info)

            # inject backdoor
            np.random.seed(seed)
            seed += 1
            injected_indices = np.random.choice(
                np.arange(x_batch.size(0)),
                int(x_batch.size(0) * args.portion), replace=False)
            uninjected_indices = list(set(np.arange(x_batch.size(0))) - set(injected_indices))
            x_batch_re = torch.clone(x_batch)
            x_batch_gt = torch.clone(x_batch)
            # print(x_batch.dtype, x_batch_gt.dtype, x_batch_re.dtype)
            # print(mask_re.dtype, pattern_re.dtype, mask_gt.dtype, pattern_gt.dtype)
            # import sys
            # sys.exit(-1)
            x_batch_re[injected_indices] = (1 - mask_re) * x_batch[injected_indices] + mask_re * pattern_re
            x_batch_gt[injected_indices] = x_batch[injected_indices] * mask_gt + pattern_gt

            x_batch[uninjected_indices] = argumentation(x_batch[uninjected_indices])
            x_batch_re[uninjected_indices] = argumentation(x_batch_re[uninjected_indices])
            x_batch_gt[uninjected_indices] = argumentation(x_batch_gt[uninjected_indices])

            x_batch_gt = torch.clip(x_batch_gt, 0.0, 1.0).cuda(args.device).float()
            x_batch_re = torch.clip(x_batch_re, 0.0, 1.0).cuda(args.device).float()
            x_batch = torch.clip(x_batch, 0.0, 1.0).cuda(args.device).float()

            x_adv_gt = preprocess(x_batch_gt, args.encoder_usage_info)
            x_adv_re = preprocess(x_batch_re, args.encoder_usage_info)
            # x_batch = new_preprocess(x_batch, args.encoder_usage_info)
            x_batch = preprocess(x_batch, args.encoder_usage_info)
            import torchvision
            torchvision.utils.save_image(
                deprocess(x_batch, clone=True, dataset=args.encoder_usage_info),
                f'{args.log_dir}/raw_img.png')
            torchvision.utils.save_image(
                deprocess(x_adv_re, clone=True, dataset=args.encoder_usage_info),
                f'{args.log_dir}/re_img.png')
            torchvision.utils.save_image(
                deprocess(x_adv_gt, clone=True, dataset=args.encoder_usage_info),
                f'{args.log_dir}/gt_img.png')
            # import torchvision
            # torchvision.utils.save_image(
            #     x_batch,
            #     f'{args.log_dir}/raw_img.png')
            # torchvision.utils.save_image(
            #     x_adv_re,
            #     f'{args.log_dir}/re_img.png')
            # torchvision.utils.save_image(
            #     x_adv_gt,
            #     f'{args.log_dir}/gt_img.png')
            # print(x_adv.dtype)
            # train model:max sim( x_adv , x_batch)
            x_batch, x_adv_re = x_batch.detach(), x_adv_re.detach()  # detach from computation graph
            x_adv_gt = x_adv_gt.detach()
            optimizer.zero_grad()
            feature_ori, out_ori = model(x_batch)
            feature_adv, out_adv = model(x_adv_re)
            # feature_adv, out_adv = model(x_adv_gt)

            loss_debackdoor = criterion_debackdoor(out_ori[injected_indices], out_adv[injected_indices])
            loss_clean = criterion_clean(out_ori[uninjected_indices], out_adv[uninjected_indices])
            # loss_debackdoor = criterion_debackdoor(feature_ori[injected_indices], feature_adv[injected_indices])
            # loss_clean = criterion_clean(feature_ori[uninjected_indices], feature_adv[uninjected_indices])
            loss = loss_clean + args.beta * loss_debackdoor
            total_loss += loss.item() * train_loader.batch_size
            total_num += train_loader.batch_size
            total_loss_clean += loss_clean.item() * train_loader.batch_size
            total_loss_debad += loss_debackdoor.item() * train_loader.batch_size
            loss.backward()
            optimizer.step()
            train_bar.set_description(
                'Train[{}/{}], Unlearn loss {:.6f}, clean loss {:.3f}, debad loss {:.3f}'
                    .format(epoch + 1, args.epochs, total_loss / total_num,
                            total_loss_clean / total_num, loss_debackdoor.item())
            )
        loss_list.append(total_loss / total_num)
        if (epoch + 1) % 1 == 0:
            torch.save(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                },
                args.results_dir + '/Harden_Unlearn_model_' + str(epoch + 1) + '.pth'
            )
        # KNN MONITOR
        if (epoch + 1) % 1 == 0:
            model.eval()
            knn_test(
                net=model.f,
                memory_data_loader=_memory_loader,
                test_data_backdoor_loader=test_backdoor_loader,
                test_data_clean_loader=test_clean_loader,
                epoch=epoch + 1,
                args=args
            )
    np.save(f'{args.log_dir}/loss-log.npy', np.array(loss_list))


def test():
    model = get_encoder_architecture_usage(args).cuda()
    if args.encoder_usage_info == 'cifar10' or args.encoder_usage_info == 'stl10':
        checkpoint = torch.load(args.pretrained_encoder)
        model.load_state_dict(checkpoint['state_dict'])

    elif args.encoder_usage_info == 'imagenet' or args.encoder_usage_info == 'CLIP':
        checkpoint = torch.load(args.pretrained_encoder)
        model.visual.load_state_dict(checkpoint['state_dict'])
    else:
        raise NotImplementedError()
    model.eval()

    test_loader = get_dataloader(args.dataset, False)

    total = 0
    correct = 0
    print('-' * 50)
    for (x_test, y_test) in test_loader:
        with torch.no_grad():
            x_test, y_test = x_test.to(args.device), y_test.to(args.device)
            total += x_test.shape[0]

            pred = model(x_test)
            correct += torch.sum(torch.argmax(pred, 1) == y_test)

            acc = correct / total

            sys.stdout.write('\racc: {:.4f}'.format(acc))
            sys.stdout.flush()
    print()


################################################################
############                  main                  ############
################################################################
def main():
    if args.phase == 'moth' and args.input_type=='poisoned_encoder':
        moth()
    elif args.phase == 'test':
        test()
    else:
        print('Option [{}] is not supported!'.format(args.phase))


def parse():
    parser = argparse.ArgumentParser(description='Process input arguments.')
    parser.add_argument('--gpu', default=2, type=int, help='gpu id')
    parser.add_argument('--seed', default=0, help='seed index', type=int)
    parser.add_argument('--results_dir', default='', help='The directory to restore our model')
    parser.add_argument('--phase', default='moth', help='phase of framework')
    parser.add_argument('--dataset', default='cifar10', help='dataset, the pretraining dataset')
    parser.add_argument('--downstreamTask', default='svhn', help='dataset, the downstream task, used to test')
    parser.add_argument('--model', default='resnet18', help='model architecture')
    parser.add_argument('--pretrained_encoder', default='clean_enc.pth', help='ckpt path')

    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--beta', default=1.0, type=float, help='the weight of clean loss and debad loss')
    parser.add_argument('--epochs', default=2, type=int, help='hardening epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--data_ratio', default=1.0, type=float, help='ratio of training samples for hardening')
    parser.add_argument('--encoder_usage_info', default='cifar10', type=str,
                        help='The dataset we use to train the pretrained encoder')
    parser.add_argument('--log_dir', default='MSE-test-loss', type=str,
                        help='To storage the mask pattern and loss')
    parser.add_argument('--reference_label', default=0, type=int,
                        help='target class in the target downstream task')
    parser.add_argument('--trigger_file', default='trigger/trigger_pt_white_21_10_ap_replace.npz', type=str,
                        help='path to the trigger file (ground truth)')
    parser.add_argument('--re_mask_pattern', default='', type=str,
                        help='path to the trigger file (inversion)')

    parser.add_argument('--knn-t', default=0.5, type=float, help='softmax temperature in kNN monitor')
    parser.add_argument('--knn-k', default=200, type=int, help='k in kNN monitor')
    parser.add_argument('--portion', default=1.0, type=float, help='the portion of samples to stamp a trigger')
    parser.add_argument('--input_type', default='poisoned_encoder', type=str,
                        help='poisoned_encoder or clean_encoder')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    # set gpu usage

    # set random seed
    SEED = [1024, 557540351, 157301989]
    SEED = SEED[args.seed]
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # set basics
    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    args.n_gpu = torch.cuda.device_count()
    args.mean, args.std = get_norm(args.dataset)
    # main function
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    # main function
    time_start = time.time()
    main()
    time_end = time.time()
    print('=' * 50)
    print('Running time:', (time_end - time_start) / 60, 'm')
    print('=' * 50)
