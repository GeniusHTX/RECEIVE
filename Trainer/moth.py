import torch
import numpy as np
import pandas as pd
from util import (
    preprocess,
    deprocess,
    argumentation
)
import copy
from scipy.stats import wasserstein_distance as wd
from DATASETS import get_dataloader, get_backdoored_loader
from EVALUATION import knn_test as test
import os
import torch.nn.functional as F
from MODELS import get_encoder_architecture_usage, EarlyStopping
import torch.nn as nn
from INVERSION import Trigger
import torchvision


# from INVERSION.inversion import Trigger
def loss_similarity(feature1, feature2):
    feature1 = F.normalize(feature1, dim=-1)
    feature2 = F.normalize(feature2, dim=-1)
    sim = - torch.sum(feature1 * feature2, dim=-1).mean()
    return sim


def contrastive_loss(feature1, feature2, knn_t):
    cur_batch_size = feature1.size(0)
    out_1 = F.normalize(feature1, dim=-1)
    out_2 = F.normalize(feature2, dim=-1)
    # [2*B, D]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / knn_t)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * cur_batch_size, device=sim_matrix.device)).bool()
    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * cur_batch_size, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / knn_t)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss


class MothTrainer:
    def __init__(self, args, trigger_steps=500):
        self.args = args
        self.model = self.load_model()
        self.reference_model = copy.deepcopy(self.model)
        self.trigger_steps = trigger_steps
        self.HEIGHT = 32
        self.WIDTH = 32
        self.CLASSES = 10
        self.trigger = Trigger(
            self.model.f, steps=trigger_steps,
            attack_succ_threshold=args.attack_threshold,
            num_classes=self.CLASSES,
            img_rows=self.HEIGHT,
            img_cols=self.WIDTH,
            init_cost=args.init_cost,
            similarity_threshold=args.similarity_threshold,
            args=args
        )
        self.eps = args.epsilon
        self.suffix_distance = 0.0
        self.early_stop = False
        self.best_idx = 0
        self.early_stopping = EarlyStopping(
            args.log_dir, args.patience, verbose=True, delta=self.eps)
        # self.trigger = Trigger(
        #     self.model.f,
        #     dataset=self.args.dataset
        # )

    def load_model(self):
        # load model
        model = get_encoder_architecture_usage(self.args)
        if self.args.encoder_usage_info == 'cifar10' or self.args.encoder_usage_info == 'stl10':
            checkpoint = torch.load(self.args.pretrained_encoder, map_location=self.args.device)
            if "f.f.0.weight" in checkpoint['state_dict'].keys():
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.f.load_state_dict(checkpoint['state_dict'])

        elif self.args.encoder_usage_info == 'imagenet' or self.args.encoder_usage_info == 'CLIP':
            checkpoint = torch.load(self.args.pretrained_encoder, map_location=self.args.device)
            model.visual.load_state_dict(checkpoint['state_dict'])
        elif self.args.encoder_usage_info == 'poisoned_encoder':
            pass
        else:
            raise NotImplementedError()
        model = model.to(self.args.device)
        # if self.args.n_gpu > 1:
        #     model = torch.nn.DataParallel(model)
        #     model = model.to(self.args.device)
        # else:
        #     model = model.to(self.args.device)
        return model

    def load_data(self):
        # memory loader is just for test on knn
        # downstream task dataset for monitors
        _memory_loader = get_dataloader(
            self.args.downstreamTask, True, 1.0,
            batch_size=self.args.batch_size,
            encoder_usage_info=self.args.encoder_usage_info
        )
        # the pretraining dataset, we remove backdoor in encoder
        re_loader = get_dataloader(
            self.args.dataset, True, self.args.data_ratio,
            batch_size=self.args.batch_size,
            encoder_usage_info=self.args.encoder_usage_info
        )
        test_backdoor_loader = get_backdoored_loader(
            self.args.downstreamTask, train=False, ratio=1.0,
            batch_size=self.args.batch_size,
            trigger_file=self.args.trigger_file,
            reference_label=self.args.reference_label,
            encoder_usage_info=self.args.encoder_usage_info
        )
        test_clean_loader = get_dataloader(self.args.downstreamTask, False, 1.0)
        return _memory_loader, re_loader, test_clean_loader, test_backdoor_loader

    def mothHarden(self):
        self.model.eval()
        TRIGGER_GT = np.load(self.args.trigger_file)
        print(self.args.device)
        mask_gt = torch.from_numpy(TRIGGER_GT['tm'].transpose(0, 3, 1, 2)).cuda(self.args.device)
        pattern_gt = torch.from_numpy(TRIGGER_GT['t'].transpose(0, 3, 1, 2)).cuda(self.args.device) / 255.0
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

        num_classes = 10
        trigger_steps = self.args.re_steps
        # num_samples = 10
        optimizer_model = torch.optim.SGD(
            self.model.parameters(), lr=self.args.harden_lr, momentum=0.9, nesterov=True)
        x_extra = y_extra = None
        for idx, (x_batch, y_batch) in enumerate(re_loader):
            if idx == 0:
                x_extra, y_extra = x_batch, y_batch
            else:
                x_extra = torch.cat((x_extra, x_batch))
                y_extra = torch.cat((y_extra, y_batch))
        print(f're data size:{x_extra.shape[0]}')
        source = target = mask = pattern = None
        # print(x_extra.shape, y_extra.shape)
        # import sys
        # sys.exit(1)
        # criterion = loss_similarity
        criterion = nn.MSELoss()
        for round_idx in range(1):
            self.model.eval()
            # self.model.train()
            loss_log = []
            x_set = x_extra.cuda(device=self.args.device)
            y_set = y_extra.cuda(device=self.args.device)
            batch_size = self.args.batch_size  # 本来是 32
            attack_size = int(self.args.attack_size)
            seed = mask_pattern_idx = 0
            for epoch in range(self.args.epochs):
                # self.model.train()
                for idx in range(x_set.shape[0] // batch_size):
                    print(f"epoch:{epoch + 1} iter: {idx + 1}/{x_set.shape[0] // batch_size}")
                    # with torch.no_grad():
                    x_batch = x_set[idx * batch_size: (idx + 1) * batch_size]
                    y_batch = y_set[idx * batch_size: (idx + 1) * batch_size]

                    # generate re batch
                    np.random.seed(seed=seed)
                    seed += 1
                    attack_indices = np.random.choice(x_set.shape[0], attack_size)
                    x_re_batch = x_set[attack_indices]
                    y_re_batch = y_set[attack_indices]

                    # shuffle attack samples
                    shuffle_indices = np.arange(attack_indices.shape[0])
                    np.random.seed(seed=seed)
                    np.random.shuffle(shuffle_indices)
                    shuffle_indices = list(shuffle_indices)
                    x_re_batch = x_re_batch[shuffle_indices]
                    y_re_batch = y_re_batch[shuffle_indices]

                    # x_batch = deprocess(x_batch, clone=True)
                    # x, y preprocessed data
                    if epoch == 0 and idx == 0:
                        mask, pattern, reference_distance, reg_best, cur_log = self.trigger.generate(
                            (source, target), x_re_batch, y_re_batch, steps=trigger_steps,
                            round_idx=round_idx, draw=False
                        )
                       
                    else:
                        init_m = mask[0].detach().cpu()  # 32 x 32
                        init_p = pattern.detach().cpu()  # 3 x 32 x 32
                        mask, pattern, reference_distance, reg_best, cur_log = self.trigger.generate(
                            (source, target),
                            x_re_batch,
                            y_re_batch,
                            init_m=init_m,
                            init_p=init_p,
                            steps=trigger_steps,
                            round_idx=round_idx,
                            learning_rate=self.args.re_lr, 
                            draw=False
                        )
                    loss_log.append(cur_log)
                    # storage mask and pattern
                    saved_mask, saved_pattern = mask.detach().cpu().numpy(), pattern.detach().cpu().numpy()
                    # print(saved_mask.shape, saved_pattern.shape) 3 x 32 x 32
                    if mask_pattern_idx % 1 == 0:
                        np.savez(f'{self.args.log_dir}/mask-pattern-{mask_pattern_idx}.npz',
                                 mask=saved_mask, pattern=saved_pattern)
                    mask_pattern_idx += 1
                    x_batch = deprocess(x_batch, clone=True, dataset=self.args.encoder_usage_info)

                    # print(mask.device, x_batch.device)

                    x_batch_adv = (1 - mask.detach()) * x_batch + mask.detach() * pattern.detach()
                    x_batch_adv = torch.clip(x_batch_adv, 0.0, 1.0)

                    x_GT_backdoor = x_batch * mask_gt + pattern_gt
                    x_GT_backdoor = torch.clip(x_GT_backdoor, 0.0, 1.0)

                    x_batch_adv = preprocess(x_batch_adv, channel_first=True, dataset=self.args.encoder_usage_info)
                    x_batch = preprocess(x_batch, channel_first=True, dataset=self.args.encoder_usage_info)
                    x_GT_backdoor = preprocess(x_GT_backdoor, clone=False,
                                               channel_first=True, dataset=self.args.encoder_usage_info) \
                        .type(torch.FloatTensor).cuda(device=self.args.device)

                    # print(x_batch.device, x_batch_adv.device, x_GT_backdoor.device)
                    # import sys
                    # sys.exit(-1)
                    x_batch_adv = x_batch_adv.detach()
                    x_batch = x_batch.detach()
                    x_GT_backdoor = x_GT_backdoor.detach()

                    self.model.train()
                    optimizer_model.zero_grad()
                    feature = self.model.f(x_batch_adv)
                    output = self.model.f(x_batch)
                    output_adv = feature
                    # loss = criterion(output, y_batch.to(DEVICE))

                    loss = criterion(output_adv, output)
                    # loss = -distance

                    # print("distance:",distance)
                    print("loss:", loss)
                    loss.backward()
                    optimizer_model.step()

                    # model.eval()
                    # test(model.f, get_dataloader(args.dataset,  True, 0.1), test_loader, args)

                    if (idx + 1) % 5 == 0:
                        self.model.eval()
                        test(
                            net=self.model.f,
                            memory_data_loader=_memory_loader,
                            test_data_backdoor_loader=test_backdoor_loader,
                            test_data_clean_loader=test_clean_loader,
                            epoch=epoch + 1,
                            args=self.args
                        )

                    # judge early stop
                    self.model.eval()
                    self.reference_model.eval()
                    x_extra_re = deprocess(x_extra, dataset=self.args.encoder_usage_info)
                    x_extra_re = x_extra_re.to(self.args.device)
                    x_extra_re = x_extra_re * (1 - mask) + mask * pattern
                    x_extra_re = preprocess(x_extra_re, dataset=self.args.encoder_usage_info)
                    x_extra = x_extra.to(self.args.device)
                    with torch.no_grad():
                        feature_re = self.reference_model.f(x_extra_re).detach().cpu()
                        feature_raw = self.reference_model.f(x_extra).detach().cpu()
                    w_distance = [wd(feature_raw[i], feature_re[i]) for i in range(feature_re.size(0))]
                    cur_distance = np.mean(w_distance)
                    print(f'{mask_pattern_idx}-distance-{cur_distance:.3f} over {self.early_stopping.best_score:.3f}')
                    self.early_stopping(cur_distance, mask_pattern_idx-1, saved_mask, saved_pattern)
                    self.early_stop = self.early_stopping.early_stop
                    if self.early_stopping.early_stop:
                        self.best_idx = self.early_stopping.best_mask_pattern_idx
                        print(f"Early stopping..."
                              f"The best distance is idx:{self.best_idx}-{self.early_stopping.best_score:.3f}")
                        break
                    # when inversion is over,record current best mask pattern
                    if (not self.early_stop) and abs(cur_distance - self.suffix_distance) < self.eps:
                        
                        # self.early_stop = True
                        self.best_idx = mask_pattern_idx - 1
                        np.savez(f'{self.args.log_dir}/mask-pattern-best.npz',
                                 mask=saved_mask, pattern=saved_pattern)
                        break
                    self.suffix_distance = cur_distance

                if self.early_stop or (epoch + 1) % self.args.epochs == 0:
                    torch.save(
                        {
                            'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                        },
                        f'{self.args.results_dir}/Harden_model_' + str(epoch + 1) + '.pth'
                    )
                    print('The best idx of reversed mask pattern: ', self.best_idx)
                    break

        # np.savez(f'{self.args.log_dir}/re_mask-pattern.npz',
        #          mask=saved_mask, pattern=saved_pattern)
        loss_log = pd.DataFrame(loss_log, columns=['loss', 'loss_sim', 'loss_reg', 'refer_sim'])
        loss_log.to_csv(f'{self.args.log_dir}/re_loss.csv', index=False)

        # model.eval()
        # mask, pattern = trigger.generate((source, target), x_extra, y_extra, steps=trigger_steps)

   
    def mothHarden_with_portion_beta(self):
        self.model.eval()
        TRIGGER_GT = np.load(self.args.trigger_file)
        print(self.args.device)
        mask_gt = torch.from_numpy(TRIGGER_GT['tm'].transpose(0, 3, 1, 2)).cuda(self.args.device)
        pattern_gt = torch.from_numpy(TRIGGER_GT['t'].transpose(0, 3, 1, 2)).cuda(self.args.device) / 255.0
        mask_gt, pattern_gt = mask_gt.float(), pattern_gt.float()
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

        num_classes = 10
        trigger_steps = self.args.re_steps
        # num_samples = 10
        optimizer_model = torch.optim.SGD(
            self.model.parameters(), lr=self.args.harden_lr, momentum=0.9, nesterov=True)
        x_extra = y_extra = None
        for idx, (x_batch, y_batch) in enumerate(re_loader):
            if idx == 0:
                x_extra, y_extra = x_batch, y_batch
            else:
                x_extra = torch.cat((x_extra, x_batch))
                y_extra = torch.cat((y_extra, y_batch))
        print(f're data size:{x_extra.shape[0]}')
        source = target = mask = pattern = None
        # print(x_extra.shape, y_extra.shape)
        # import sys
        # sys.exit(1)
        # criterion = loss_similarity
        # criterion = nn.MSELoss()
        criterion_debackdoor = nn.MSELoss()
        # criterion_debackdoor = loss_similarity
        criterion_clean = contrastive_loss
        for round_idx in range(1):
            self.model.eval()
            # self.model.train()
            loss_log = []  # reverse loss
            harden_loss_log = []
            x_set = x_extra.cuda(device=self.args.device)
            y_set = y_extra.cuda(device=self.args.device)
            batch_size = self.args.batch_size  # 本来是 32
            attack_size = int(self.args.attack_size)
            seed = mask_pattern_idx = 0
            for epoch in range(self.args.epochs):
                # self.model.train()
                total_loss, total_loss_debad, total_loss_clean = 0.0, 0.0, 0.0
                for idx in range(x_set.shape[0] // batch_size):
                    print(f"epoch:{epoch + 1} iter: {idx + 1}/{x_set.shape[0] // batch_size}")
                    # with torch.no_grad():
                    x_batch = x_set[idx * batch_size: (idx + 1) * batch_size]
                    y_batch = y_set[idx * batch_size: (idx + 1) * batch_size]

                    # generate re batch
                    np.random.seed(seed=seed)
                    seed += 1
                    attack_indices = np.random.choice(x_set.shape[0], attack_size)
                    x_re_batch = x_set[attack_indices]
                    y_re_batch = y_set[attack_indices]

                    # shuffle attack samples
                    shuffle_indices = np.arange(attack_indices.shape[0])
                    np.random.seed(seed=seed)
                    np.random.shuffle(shuffle_indices)
                    shuffle_indices = list(shuffle_indices)
                    x_re_batch = x_re_batch[shuffle_indices]
                    y_re_batch = y_re_batch[shuffle_indices]

                    # x_batch = deprocess(x_batch, clone=True)
                    # x, y preprocessed data
                    if epoch == 0 and idx == 0:
                        mask, pattern, reference_distance, reg_best, cur_log = self.trigger.generate(
                            (source, target), x_re_batch, y_re_batch, steps=trigger_steps,
                            round_idx=round_idx, draw=False
                        )
                    else:
                        init_m = mask[0].detach().cpu()  # 32 x 32
                        init_p = pattern.detach().cpu()  # 3 x 32 x 32
                        mask, pattern, reference_distance, reg_best, cur_log = self.trigger.generate(
                            (source, target),
                            x_re_batch,
                            y_re_batch,
                            init_m=init_m,
                            init_p=init_p,
                            steps=trigger_steps,
                            round_idx=round_idx,
                            learning_rate=self.args.re_lr,
                            draw=False
                        )
                    loss_log.append(cur_log)
                    # storage mask and pattern
                    saved_mask, saved_pattern = mask.detach().cpu().numpy(), pattern.detach().cpu().numpy()
                    # print(saved_mask.shape, saved_pattern.shape) 3 x 32 x 32
                    if mask_pattern_idx % 5 == 0:
                        np.savez(f'{self.args.log_dir}/mask-pattern-{mask_pattern_idx}.npz',
                                 mask=saved_mask, pattern=saved_pattern)
                    mask_pattern_idx += 1

                    # init raw x_batch, x_batch_adv, x_batch_gt
                    x_batch = deprocess(x_batch, clone=True, dataset=self.args.encoder_usage_info)
                    x_batch_re = torch.clone(x_batch)
                    x_batch_gt = torch.clone(x_batch)

                    # inject backdoor
                    np.random.seed(seed)
                    seed += 1
                    injected_indices = np.random.choice(
                        np.arange(x_batch.size(0)),
                        int(x_batch.size(0) * self.args.portion), replace=False)
                    uninjected_indices = list(set(np.arange(x_batch.size(0))) - set(injected_indices))

                    # 根据指定的 portion inject mask & trigger
                    mask_re, pattern_re = mask.detach(), pattern.detach()
                    x_batch_re[injected_indices] = (1 - mask_re) * x_batch[injected_indices] + mask_re * pattern_re
                    x_batch_gt[injected_indices] = x_batch[injected_indices] * mask_gt + pattern_gt

                    x_batch[uninjected_indices] = argumentation(x_batch[uninjected_indices])
                    x_batch_re[uninjected_indices] = argumentation(x_batch_re[uninjected_indices])
                    x_batch_gt[uninjected_indices] = argumentation(x_batch_gt[uninjected_indices])

                    x_batch_gt = torch.clip(x_batch_gt, 0.0, 1.0).cuda(self.args.device).float()
                    x_batch_re = torch.clip(x_batch_re, 0.0, 1.0).cuda(self.args.device).float()
                    x_batch = torch.clip(x_batch, 0.0, 1.0).cuda(self.args.device).float()

                    x_adv_gt = preprocess(x_batch_gt, self.args.encoder_usage_info)
                    x_adv_re = preprocess(x_batch_re, self.args.encoder_usage_info)
                    # x_batch = new_preprocess(x_batch, args.encoder_usage_info)
                    x_batch = preprocess(x_batch, self.args.encoder_usage_info)
                    torchvision.utils.save_image(
                        deprocess(x_batch, clone=True, dataset=self.args.encoder_usage_info),
                        f'{self.args.log_dir}/raw_img.png')
                    torchvision.utils.save_image(
                        deprocess(x_adv_re, clone=True, dataset=self.args.encoder_usage_info),
                        f'{self.args.log_dir}/re_img.png')
                    torchvision.utils.save_image(
                        deprocess(x_adv_gt, clone=True, dataset=self.args.encoder_usage_info),
                        f'{self.args.log_dir}/gt_img.png')

                    self.model.train()
                    x_batch, x_adv_re = x_batch.detach(), x_adv_re.detach()  
                    x_adv_gt = x_adv_gt.detach()
                    optimizer_model.zero_grad()
                    feature_ori, out_ori = self.model(x_batch)
                    feature_adv, out_adv = self.model(x_adv_re)
                    # feature_adv, out_adv = model(x_adv_gt)

                    loss_debackdoor = criterion_debackdoor(out_ori[injected_indices], out_adv[injected_indices])
                    loss_clean = criterion_clean(
                        out_ori[uninjected_indices], out_adv[uninjected_indices], self.args.knn_t
                    )
                    loss = loss_clean + self.args.beta * loss_debackdoor
                    # loss_batch.append([loss.item(), loss_clean.item(), loss_debackdoor.item(), self.args.beta])
                    total_loss += loss.item()
                    total_loss_clean += loss_clean.item()
                    total_loss_debad += loss_debackdoor.item()
                    loss.backward()
                    optimizer_model.step()

                    print('\nUnlearn loss {:.6f}, clean loss {:.3f}, debad loss {:.3f}'
                          .format(loss.item(), loss_clean.item(), loss_debackdoor.item()))

                    # model.eval()
                    # test(model.f, get_dataloader(args.dataset,  True, 0.1), test_loader, args)

                    if (idx + 1) % 5 == 0:
                        self.model.eval()
                        test(
                            net=self.model.f,
                            memory_data_loader=_memory_loader,
                            test_data_backdoor_loader=test_backdoor_loader,
                            test_data_clean_loader=test_clean_loader,
                            epoch=epoch + 1,
                            args=self.args
                        )

                harden_loss_log.append([total_loss, total_loss_clean, total_loss_debad, self.args.beta])

                if (epoch + 1) % 1 == 0:
                    torch.save(
                        {
                            'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                        },
                        f'{self.args.results_dir}/Harden_model_' + str(epoch + 1) + '.pth'
                    )

        # save loss
        harden_loss_log = pd.DataFrame(harden_loss_log,
                                       columns=['loss', 'clean_loss', 'debad_loss', 'beta'])
        harden_loss_log.to_csv(f'{self.args.log_dir}/harden_loss.csv', index=False)
        np.savez(f'{self.args.log_dir}/re_mask-pattern.npz',
                 mask=saved_mask, pattern=saved_pattern)
        loss_log = pd.DataFrame(loss_log, columns=['loss', 'loss_sim', 'loss_reg', 'reference_sim'])
        loss_log.to_csv(f'{self.args.log_dir}/re_loss.csv', index=False)

        # model.eval()
        # mask, pattern = trigger.generate((source, target), x_extra, y_extra, steps=trigger_steps)
