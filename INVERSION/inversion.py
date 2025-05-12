import torch
import numpy as np
from util import (
    deprocess,
    preprocess
)
from DATASETS import get_dataloader
import torchvision
import torch.nn.functional as F
import sys


class Trigger:
    def __init__(self, model, args,
                 batch_size=32, steps=1000,
                 img_rows=224, img_cols=224, img_channels=3,
                 num_classes=10,
                 attack_succ_threshold=0.9,
                 similarity_threshold=-0.9, init_cost=1e-3):
        self.args = args
        self.model = model
        self.batch_size = batch_size
        self.steps = steps
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.attack_succ_threshold = attack_succ_threshold
        self.similarity_threshold = similarity_threshold
        self.init_cost = init_cost
        self.seed = 100
        self.device = self.args.device
        self.epsilon = 1e-7
        self.patience = 5
        # self.cost_multiplier_up = 3
        self.cost_multiplier_up = 1.2 # only for stl10 - svhn
        # self.cost_multiplier_up = 5.0
        self.cost_multiplier_down = 1.5 ** 1.5

        self.mask_size = [self.img_rows, self.img_cols]
        self.pattern_size = [self.img_channels, self.img_rows, self.img_cols]

    def generate(self, pair, x_train, y_train,
                 attack_size=32, steps=1000,
                 learning_rate=0.5, init_m=None, init_p=None,
                 round_idx=0, tsne_img_name=None, draw_mask=None,
                 draw_pattern=None, draw=True, draw_tsne=False):
        self.model.eval()
        cur_loss_return = []
        cur_loss_reg = []
        cur_loss_sim = []
        self.steps = steps
        source, target = pair
        cost = self.init_cost
        cost_up_counter = 0
        cost_down_counter = 0

        dis_adv2clean_threshold = 20
        TRIGGER_GT = np.load("trigger/trigger_pt_white_21_10_ap_replace.npz")
        # print(TRIGGER_GT['t'].shape)
        # print(TRIGGER_GT['tm'].shape)
        mask_gt = torch.from_numpy(TRIGGER_GT['tm'].transpose(0, 3, 1, 2).astype(np.float32)).cuda(self.device)
        pattern_gt = torch.from_numpy(TRIGGER_GT['t'].transpose(0, 3, 1, 2).astype(np.float32)).cuda(self.device) / 255
        mask_best = torch.zeros(self.pattern_size).to(self.device)
        pattern_best = torch.zeros(self.pattern_size).to(self.device)
        reg_best = float('inf')

        if init_m is None:
            init_mask = np.random.random(self.mask_size)
        else:
            init_mask = init_m
        if init_p is None:
            init_pattern = np.random.random(self.pattern_size)
        else:
            init_pattern = init_p
        init_mask = np.clip(init_mask, 0.0, 1.0)
        init_mask = np.arctanh((init_mask - 0.5) * (2 - self.epsilon))
        init_pattern = np.clip(init_pattern, 0.0, 1.0)
        init_pattern = np.arctanh((init_pattern - 0.5) * (2 - self.epsilon))

        self.mask_tensor = torch.Tensor(init_mask).to(self.device)
        self.pattern_tensor = torch.Tensor(init_pattern).to(self.device)
        self.mask_tensor.requires_grad = True
        self.pattern_tensor.requires_grad = True

        if source is not None:
            indices = np.where(y_train == source)[0]
            if indices.shape[0] > attack_size:
                indices = np.random.choice(indices, attack_size, replace=False)
            else:
                attack_size = indices.shape[0]

            if attack_size < self.batch_size:
                self.batch_size = attack_size

            x_set = x_train[indices]
            y_set = torch.full((x_set.shape[0],), target)
        else:
            x_set, y_set = x_train, y_train
            source = self.num_classes
            self.batch_size = attack_size
            loss_start = np.zeros(x_set.shape[0])
            loss_end = np.zeros(x_set.shape[0])
        x_set = x_set.to(self.device)
        y_set = y_set.to(self.device)

        self.model.eval()
        with torch.no_grad():
            feature_bank, feature_bank_clean = [], []
            for idx in range(x_set.shape[0] // self.batch_size):
                x_batch = x_set[idx * self.batch_size: (idx + 1) * self.batch_size]
                x_batch = x_batch.to(self.args.device)
                x_gt_batch = deprocess(x_batch, dataset=self.args.encoder_usage_info).to(
                    self.args.device) * mask_gt + pattern_gt
                feature = self.model(preprocess(x_gt_batch, self.args.encoder_usage_info))
                feature_clean = self.model(preprocess(x_batch, self.args.encoder_usage_info))
                feature_bank.append(feature)
                feature_bank_clean.append(feature_clean)
            feature_bank = torch.cat(feature_bank, dim=0)
            sim_matrix = torch.mm(F.normalize(feature_bank, dim=-1), F.normalize(feature_bank, dim=-1).T)
            reference_similarity = -(sim_matrix - torch.diag_embed(sim_matrix)).mean()
            print("refer_sim_gt: ", reference_similarity.item())

            feature_bank_clean = torch.cat(feature_bank_clean, dim=0)
            sim_matrix = torch.mm(F.normalize(feature_bank_clean, dim=-1), F.normalize(feature_bank_clean, dim=-1).T)
            reference_similarity = -(sim_matrix - torch.diag_embed(sim_matrix)).mean()
            print("refer_sim_clean: ", reference_similarity.item())

        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam([self.mask_tensor, self.pattern_tensor], lr=learning_rate, betas=(0.5, 0.9))

        index_base = np.arange(x_set.shape[0])
        # for step in range(self.steps):
        step = 0
        while (step < self.steps) or (reg_best == float('inf')):
            # print("step:",step)
            # 仅仅shuffle 一下
            indices = np.arange(x_set.shape[0])
            np.random.seed(self.seed)
            self.seed += 1
            np.random.shuffle(indices)
            index_base = index_base[indices]
            x_set = x_set[indices]
            y_set = y_set[indices]

            loss_reg_list = []
            loss_list = []
            similarity_list = []

            for idx in range(x_set.shape[0] // self.batch_size):
                x_batch = x_set[idx * self.batch_size: (idx + 1) * self.batch_size]
                y_batch = y_set[idx * self.batch_size: (idx + 1) * self.batch_size]

                # deprocessed
                x_batch = deprocess(x_batch, clone=False, dataset=self.args.encoder_usage_info)

                # define mask and pattern
                self.mask = (torch.tanh(self.mask_tensor) / (2 - self.epsilon) + 0.5) \
                    .repeat(self.img_channels, 1, 1)
                self.pattern = (torch.tanh(self.pattern_tensor) / (2 - self.epsilon) + 0.5)

                x_adv = (1 - self.mask) * x_batch + self.mask * self.pattern
                x_adv = torch.clip(x_adv, 0.0, 1.0)
                optimizer.zero_grad()

                feature = self.model(
                    preprocess(x_adv, clone=False, channel_first=True, dataset=self.args.encoder_usage_info).to(
                        self.args.device)
                )

                x_GT_backdoor = x_batch * mask_gt + pattern_gt

                sim_matrix = torch.mm(F.normalize(feature, dim=-1), F.normalize(feature, dim=-1).T)
                similarity = -(sim_matrix - torch.diag_embed(sim_matrix)).mean()

                loss_reg = torch.sum(torch.abs(self.mask)) / self.img_channels

                loss = similarity + loss_reg * cost
                # print(f'ini:{similarity.item()}, {loss_reg}, {cost}')
                loss.backward()
                optimizer.step()
                loss_reg_list.append(loss_reg.detach().cpu().numpy())
                loss_list.append(loss.detach().cpu().numpy())
                similarity_list.append(similarity.detach().cpu().numpy())

            avg_loss_reg = np.mean(loss_reg_list)
            avg_loss = np.mean(loss_list)
            avg_similarity = np.mean(similarity_list)
            cur_loss_reg.append(avg_loss_reg)
            cur_loss_return.append(avg_loss)
            cur_loss_sim.append(avg_similarity)

            torchvision.utils.save_image(x_adv, f"{self.args.log_dir}/re_imgs.jpg")
            torchvision.utils.save_image(x_batch, f"{self.args.log_dir}/ori_imgs.jpg")

            torchvision.utils.save_image(x_GT_backdoor, f"{self.args.log_dir}/gt_backdoor_imgs.jpg")
            import matplotlib.pyplot as plt
            # print(x_adv.shape)
            # import sys
            # sys.exit(-1)

            # if avg_similarity <= self.similarity_threshold and avg_loss_reg < reg_best and dis_adv2clean > (
            #         dis_adv2clean_threshold * 0.95):
            # record best mask and pattern
            if abs(avg_similarity) >= abs(self.similarity_threshold) and avg_loss_reg < reg_best:
                # avoid local minimal
                mask_best = self.mask
                pattern_best = self.pattern
                reg_best = avg_loss_reg
                epsilon = 0.01
                init_mask = mask_best[0, ...]
                torch.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
                init_mask = init_mask + torch.distributions.Uniform(low=-epsilon, high=epsilon) \
                    .sample(init_mask.shape).to(self.device)
                init_mask = torch.clip(init_mask, 0.0, 1.0)
                init_mask = torch.arctanh((init_mask - 0.5) * (2 - self.epsilon))
                torch.manual_seed(self.seed + 1)
                torch.cuda.manual_seed_all(self.seed + 1)
                init_pattern = pattern_best + torch.distributions.Uniform(low=-epsilon, high=epsilon) \
                    .sample(init_pattern.shape).to(self.device)
                init_pattern = torch.clip(init_pattern, 0.0, 1.0)
                init_pattern = torch.arctanh((init_pattern - 0.5) * (2 - self.epsilon))
                # self.seed += 1

                with torch.no_grad():
                    self.mask_tensor.copy_(init_mask)
                    self.pattern_tensor.copy_(init_pattern)

            # adjust weight
            if abs(avg_similarity) >= abs(self.similarity_threshold):
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            # adjust loss weight
            if cost_up_counter >= self.patience:
                cost_up_counter = 0
                if cost == 0:
                    cost = self.init_cost
                else:
                    cost *= self.cost_multiplier_up
            elif cost_down_counter >= self.patience:
                cost_down_counter = 0
                cost /= self.cost_multiplier_down

            if step % 10 == 0:
                import sys
                sys.stdout.write('\rstep: %3d, similarity: %.3f, loss: %f, reg: %f, reg_best: %f, cost: %f' %
                                 (step, avg_similarity, avg_loss, avg_loss_reg, reg_best, cost))
                sys.stdout.flush()
                # print('\rstep: %3d, similarity: %.3f, loss: %f, reg: %f, reg_best: %f' %
                #       (step, avg_similarity, avg_loss, avg_loss_reg, reg_best))

                # img = x_adv.detach().cpu().numpy().transpose(0, 2, 3, 1)[0]
                # # print(img.shape)
                # plt.imshow(img)
                # plt.show()
                # plt.close()
            with torch.no_grad():
                mask_, pattern_ = self.mask.cpu().numpy(), self.pattern.cpu().numpy()
                np.savez(f'{self.args.log_dir}/re_new.npz', mask=mask_, pattern=pattern_)
            step = step + 1

        reference_distance = 0
        # sys.exit(-1)
        return mask_best, pattern_best, reference_distance, reg_best,\
               [np.mean(cur_loss_return), np.mean(cur_loss_sim), np.mean(cur_loss_reg), reference_similarity.item()]
