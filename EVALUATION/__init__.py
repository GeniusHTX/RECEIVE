import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn_classifier import NeuralNet, create_torch_dataloader, net_train, net_test
from .nn_classifier import predict_feature


def knn_test(net, memory_data_loader, test_data_clean_loader, test_data_backdoor_loader, epoch, args):
    net.eval()
    dataset = args.downstreamTask if hasattr(args, 'downstreamTask') else args.dataset
    if dataset == 'svhn':
        classes = 10
    else:
        classes = len(memory_data_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank, labels_bank = 0.0, 0.0, 0, [], []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature = net(data.cuda(non_blocking=True, device=args.device))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            labels_bank.append(target.cuda(non_blocking=True, device=args.device))
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        # feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        feature_labels = torch.cat(labels_bank, dim=0)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_clean_loader)
        for data, target in test_bar:
            data = data.cuda(non_blocking=True, device=args.device)
            target = target.cuda(non_blocking=True, device=args.device)
            feature = net(data)
            feature = F.normalize(feature, dim=1)
            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, args.knn_k, args.knn_t)
            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_description(
                'Test Epoch: [{}/{}] Acc@clean:{:.2f}%'.format(epoch, args.epochs, total_top1 / total_num * 100))

        total_num, total_top1 = 0., 0.
        test_bar = tqdm(test_data_backdoor_loader)
        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True, device=args.device), target.cuda(non_blocking=True, device=args.device)
            feature = net(data)
            feature = F.normalize(feature, dim=1)
            # import torchvision
            # torchvision.utils.save_image(data, 'backdoored img.png')
            # import sys
            # sys.exit(-1)
            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, args.knn_k, args.knn_t)
            # print(pred_labels[:, 0])
            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_description(
                'Test Epoch: [{}/{}] ASR@back_:{:.2f}%'.format(epoch, args.epochs, total_top1 / total_num * 100))
    return total_top1 / total_num * 100


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


