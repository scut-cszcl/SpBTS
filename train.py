# -*- coding: utf-8 -*-
from torchvision.utils import save_image
import torch
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
from torch.backends import cudnn
import numpy as np
import os
import time
import datetime
import argparse
import random
import math
import json
import cv2

from data.brain import get_loaders
from utils.util import dice_score, check_dirs, print_net, compute_ssim
from net.SpBTS import Discriminator, Content_Discriminator, ST
from scipy import ndimage
from process.utils import parse_image_name

class Solver:
    def __init__(self, data_files, opt):
        self.opt = opt
        self.best_epoch = 0
        self.best_dice = 0
        self.best_epoch_extra = 0
        self.best_dice_extra = 0

        # Data Loader.
        self.phase = self.opt.phase
        self.selected_modal = self.opt.selected_modal
        self.image_size = self.opt.image_size
        self.batch_size = self.opt.batch_size
        self.num_workers = self.opt.num_workers
        self.seg_type = self.opt.seg_type
        loaders = get_loaders(data_files, self.selected_modal, self.batch_size,
                              self.num_workers, self.image_size, self.seg_type)
        self.loaders = {x: loaders[x] for x in ('train_pair', 'train_unpair', 'val', 'test')}

        # Model Configurations.
        self.c_dim = len(self.selected_modal)
        self.in_channels = self.c_dim + 1
        self.out_channels = self.opt.out_channels
        self.content_channel = self.opt.content_channel
        self.feature_maps = self.opt.feature_maps
        self.levels = self.opt.levels
        self.nz = self.opt.nz
        self.norm_type = self.opt.norm_type
        self.use_dropout = self.opt.use_dropout
        self.d_conv_dim = self.opt.d_conv_dim
        self.d_repeat_num = self.opt.d_repeat_num
        self.wskip = self.opt.wskip
        self.wtrans = self.opt.wtrans
        self.wlscon = self.opt.wlscon
        self.wlstran = self.opt.wlstran
        self.save_rf = self.opt.save_rf

        self.lambda_cls = self.opt.lambda_cls
        self.lambda_rec = self.opt.lambda_rec
        self.lambda_gp = self.opt.lambda_gp
        self.lambda_seg = self.opt.lambda_seg
        self.lambda_trans = self.opt.lambda_trans
        self.lambda_real = self.opt.lambda_real
        self.lambda_fake = self.opt.lambda_fake
        self.lambda_con = self.opt.lambda_con
        self.lambda_shape = self.opt.lambda_shape
        self.lambda_cadv = self.opt.lambda_cadv
        self.lambda_kl = self.opt.lambda_kl
        self.lambda_sty = self.opt.lambda_sty

        # Train Configurations.
        self.max_epoch = self.opt.max_epoch
        # self.power = self.opt.power
        self.decay_epoch = self.opt.decay_epoch
        self.furthertrain = self.opt.furthertrain
        self.furthertrain_epoch = self.opt.furthertrain_epoch
        self.pretrain_epoch = self.opt.pretrain_epoch
        self.g_lr = self.opt.g_lr
        self.d_lr = self.opt.d_lr
        self.min_g_lr = self.opt.min_g_lr
        self.min_d_lr = self.opt.min_d_lr
        self.beta1 = self.opt.beta1
        self.beta2 = self.opt.beta2
        self.ignore_index = self.opt.ignore_index
        self.seg_loss_type = self.opt.seg_loss_type
        self.n_critic = self.opt.n_critic

        # Test Configurations.
        self.test_epoch = self.opt.test_epoch

        # Miscellaneous
        self.use_tensorboard = self.opt.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.checkpoint_dir = self.opt.checkpoint_dir
        self.log_dir = os.path.join(self.checkpoint_dir, 'logs')
        self.sample_dir = os.path.join(self.checkpoint_dir, 'sample_dir')
        self.model_save_dir = os.path.join(self.checkpoint_dir, 'model_save_dir')
        self.result_dir = os.path.join(self.checkpoint_dir, 'result_dir')
        self.generation_dir = self.checkpoint_dir + '/generation'
        self.sty_json_dir = self.checkpoint_dir + '/styimage/json/sty.json'
        self.ssim_json_dir = self.checkpoint_dir + '/styimage/json/ssim.json'
        self.save_sty_image = self.checkpoint_dir + '/styimage'
        self.save_sty_image_map = self.checkpoint_dir + '/styimage/map'

        check_dirs([self.log_dir, self.sample_dir, self.model_save_dir, self.result_dir, self.generation_dir, self.save_sty_image, self.save_sty_image_map])
        check_dirs(self.checkpoint_dir + '/styimage/json')
        for m in self.selected_modal:
            check_dirs(os.path.join(self.generation_dir, m))

        # Step Size.
        self.log_step = self.opt.log_step
        self.val_epoch = self.opt.val_epoch
        self.lr_update_epoch = self.opt.lr_update_epoch
        # self.idxs = torch.tensor([[0, 1], [2, 3]]).to(self.device)

        # Build Model and Tensorboard.
        self.G = None
        self.D = None
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        self.G = ST(1, self.out_channels, self.in_channels, 1, self.feature_maps, self.levels, self.nz, self.norm_type,
                    self.use_dropout, self.wskip, self.wtrans)
        if self.phase == 'train':
            print_net(self.G)
        self.G.to(self.device)

        if self.phase != 'score':
            if self.wtrans:
                self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)
                print_net(self.D)
                self.D.to(self.device)
                self.d_optimizer = optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2], weight_decay=0.0001)



            self.g_optimizer = optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2], weight_decay=0.0001)


    def build_tensorboard(self):
        from tensorboardX import SummaryWriter
        self.writer = SummaryWriter(self.log_dir)

    def restore_model(self, resume_epoch):
        print('Resume the trained models from step {}...'.format(resume_epoch))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_epoch))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        if self.wtrans:
            D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_epoch))
            self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))


    def save_model(self, save_iters):
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(save_iters))
        torch.save(self.G.state_dict(), G_path)
        if self.wtrans:
            D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(save_iters))
            torch.save(self.D.state_dict(), D_path)

        print('Saved model checkpoints into {}...'.format(self.model_save_dir))

    def update_lr(self, g_lr, d_lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        if self.wtrans:
            for param_group in self.d_optimizer.param_groups:
                param_group['lr'] = d_lr


    def reset_grad(self):
        self.g_optimizer.zero_grad()
        if self.wtrans:
            self.d_optimizer.zero_grad()

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def get_distance(self, f):
        """Return the signed distance."""
        threshold = 8
        f = np.array(f.cpu())
        dist_func = ndimage.distance_transform_edt
        distance = np.where(f, -(dist_func(f)),
                            dist_func(1 - f))
        new_label = torch.from_numpy(distance).long().to(self.device)
        new_label[new_label <= threshold] = 1
        new_label[new_label > threshold] = 0
        return new_label.float()

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def label2onenot(self, labels, dim):
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    @staticmethod
    def classification_loss(logit, target):
        return F.cross_entropy(logit, target)

    def seg_loss(self, outputs, targets, ignore_idx, loss_type='cross-entropy', weight=None):
        if loss_type == 'cross-entropy':
            if weight is not None:
                weight = weight.to(self.device)
                if ignore_idx:
                    return F.cross_entropy(outputs, targets, ignore_index=ignore_idx, weight=weight)
                return F.cross_entropy(outputs, targets, weight=weight)
            if ignore_idx:
                return F.cross_entropy(outputs, targets, ignore_index=ignore_idx)
            return F.cross_entropy(outputs, targets)
        else:
            exit('[Error] Loss type not found!')

    def dice_loss(self, outputs, targets):
        eps = 1e-7

        inse = (outputs * targets).sum().float()
        l = (outputs * outputs).sum().float()
        r = targets.sum().float()
        dice = 2.0 * inse / (l + r + eps)
        return 1.0 - 1.0 * dice

    # def TF_loss(self, out, type='fake'):
    #    out = torch.sigmoid(out)
    #    if type == 'fake':
    #        all0 = torch.zeros_like(out).to(self.device)
    #    else:
    #        all0 = torch.ones_like(out).to(self.device)
    #    return nn.functional.binary_cross_entropy(out, all0)

    def TF_loss(self, out, type='fake'):
        if type == 'fake':
            return torch.mean(out)
        else:
            return - torch.mean(out)

    def trans_adv_loss(self, image, label_org, rec, fake_image, flag_pair):
        if not self.wtrans:
            return 0.0, 0.0, 0.0, 0.0
        out_real_src1, out_cls1 = self.D(image)
        d_loss_real = self.TF_loss(out_real_src1, 'real')
        d_loss_cls = self.classification_loss(out_cls1, label_org)

        rout_fake_src1, _ = self.D(rec.detach())
        if flag_pair:
            out_fake_src1, _ = self.D(fake_image.detach())
            d_loss_fake = 0.5 * self.TF_loss(out_fake_src1, 'fake') + \
                           0.5 * self.TF_loss(rout_fake_src1, 'fake')
        else:
            d_loss_fake = self.TF_loss(rout_fake_src1, 'fake')

        # Compute loss for gradient penalty.
        alpha1 = torch.rand(image.size(0), 1, 1, 1).to(self.device)
        rx_hat1 = (alpha1 * image.data + (1 - alpha1) * rec.data).requires_grad_(True)
        rout_src1, _ = self.D(rx_hat1)
        if flag_pair:
            x_hat1 = (alpha1 * image.data + (1 - alpha1) * fake_image.data).requires_grad_(True)
            out_src1, _ = self.D(x_hat1)
            d_loss_gp = 0.5 * self.gradient_penalty(out_src1, x_hat1) + \
                         0.5 * self.gradient_penalty(rout_src1, rx_hat1)
        else:
            d_loss_gp = 0.5 * self.gradient_penalty(rout_src1, rx_hat1)
        return d_loss_real, d_loss_cls, d_loss_fake, d_loss_gp

    def cadv_loss(self, content, vec, type='D'):
        # if type == 'D':
        #     out_cls = torch.sigmoid(self.cD(content.detach()))
        #     d_loss_cadv = F.binary_cross_entropy(out_cls, vec)
        # elif type == 'G':
        #     out_cls = torch.sigmoid(self.cD(content))
        #     lable_mediocre = 0.5 * torch.ones_like(out_cls).to(self.device)
        #     d_loss_cadv = F.binary_cross_entropy(out_cls, lable_mediocre)
        # else:
        #     assert 1 > 2, 'error cadv type'
        # return d_loss_cadv
        return torch.tensor(0.0)

    def sty_loss(self, sty):
        mu_2 = torch.pow(sty, 2)
        skl_loss = torch.mean(mu_2) * 0.01
        return skl_loss


    def seg_adv_loss(self, image_s, label_s, fake_image_s2t, rec_s, label_org_s, image_t, label_org_t, train_type):
        if not self.wtrans:
            return 0.0, 0.0, 0.0, 0.0
        out_src1, out_cls1 = self.D(fake_image_s2t)
        rout_src1, rout_cls1 = self.D(rec_s)
        g_loss_fake1 = 0.5 * self.TF_loss(out_src1, 'real') + \
                       0.5 * self.TF_loss(rout_src1, 'real')
        g_loss_cls1 = 0.5 * self.classification_loss(out_cls1, label_org_t) + \
                      0.5 * self.classification_loss(rout_cls1, label_org_s)

        # new_label = self.get_distance(label_s).detach()
        g_loss_rec1 = torch.mean(torch.abs(image_s - rec_s))
        # g_loss_rec1 = 0.6*torch.mean(torch.abs(image_s - rec_s)) \
        #               + 0.4*torch.mean(torch.abs(image_s*new_label - rec_s*new_label))
        if self.wlstran:
            if train_type == 2:
                g_loss_trans1 = torch.mean(torch.abs(image_t - fake_image_s2t))
                # g_loss_trans1 = torch.mean(torch.abs(image_t - fake_image_s2t)) + \
                #                 torch.mean(torch.abs(image_t*new_label - fake_image_s2t*new_label))

            else:
                g_loss_trans1 = torch.mean(torch.abs(image_s - fake_image_s2t))
                # g_loss_trans1 = 0.6*torch.mean(torch.abs(image_s - fake_image_s2t)) + \
                #                 0.4*torch.mean(torch.abs(image_s*new_label - fake_image_s2t*new_label))
        else:
            g_loss_trans1 = 0.0
        #if train_type != 2:
        #    return g_loss_fake1, g_loss_cls1, g_loss_rec1, g_loss_trans1
        #else:
        #    return g_loss_fake1, g_loss_cls1, g_loss_rec1, torch.tensor(0.0)
        return g_loss_fake1, g_loss_cls1, g_loss_rec1, g_loss_trans1
        # return g_loss_fake1, g_loss_cls1, torch.tensor(0.0), g_loss_trans1

    def seg_loss_base(self, dseg, label_s):
        label = label_s.unsqueeze(dim=1)
        g_loss_seg = None
        for i in range(3):
            if g_loss_seg is None:
                g_loss_seg = 0.5 * self.dice_loss(dseg[i].softmax(dim=1)[:, 1, :, :], F.interpolate(label.float(), scale_factor=0.5**i, mode='nearest').squeeze(dim=1).detach())
            else:
                g_loss_seg += 0.25 * self.dice_loss(dseg[i].softmax(dim=1)[:, 1, :, :], F.interpolate(label.float(), scale_factor=0.5**i, mode='nearest').squeeze(dim=1).detach())
        return g_loss_seg

    def seg_seg_loss(self, dseg, label, content1, content2, weight):
        seg = dseg[0]
        _, pred_x1 = torch.max(seg, 1)
        cur_dices = dice_score(pred_x1, label.detach(), 1, reduce=True) * label.size(0)
        cur_samples = label.size(0)
        ## g_loss_seg = 0.5 * self.seg_loss(seg, label.detach(), self.ignore_index,
        ##                             self.seg_loss_type,
        ##                             weight)
        ## g_loss_seg += 0.5 * self.dice_loss(seg.softmax(dim=1)[:, 1, :, :], label.detach())
        g_loss_seg = self.dice_loss(seg.softmax(dim=1)[:, 1, :, :], label.detach())
        # g_loss_seg = self.seg_loss_base(dseg, label)
        if self.wlscon:
            # g_loss_con = torch.abs(torch.mean(torch.abs(content1 - content2)) - 0.2)
            g_loss_con = torch.mean(torch.abs(content1 - content2))
        else:
            g_loss_con = 0.0
        return cur_dices, cur_samples, g_loss_seg, g_loss_con
        # return cur_dices, cur_samples, g_loss_seg, torch.tensor(0.0)

    def unpair_seg_loss(self, image, dseg_score, rec, fake, label, label_org, label_tar, weight):
        seg_score = dseg_score[0]
        _, pred = torch.max(seg_score, 1)
        cur_dices = dice_score(pred, label.detach(), 1, reduce=True) * label.size(0)
        cur_samples = label.size(0)
        # g_loss_seg = 0.5 * self.seg_loss(seg_score, label.detach(), self.ignore_index,
        #                            self.seg_loss_type,
        #                            weight)
        # g_loss_seg += 0.5 * self.dice_loss(seg_score.softmax(dim=1)[:, 1, :, :], label.detach())
        g_loss_seg = self.dice_loss(seg_score.softmax(dim=1)[:, 1, :, :], label.detach())
        # g_loss_seg = self.seg_loss_base(dseg_score, label)
        if not self.wtrans:
            return cur_dices, cur_samples, g_loss_seg, 0.0, 0.0, 0.0

        rout_src, rout_cls = self.D(rec)
        out_src, out_cls = self.D(fake)
        # g_loss_fake = 0.5 * self.TF_loss(rout_src, 'real') + \
        #               0.5 * self.TF_loss(out_src, 'real')
        #
        # g_loss_cls = 0.5 * self.classification_loss(rout_cls, label_org) + \
        #              0.5 * self.classification_loss(out_cls, label_tar)

        g_loss_fake = self.TF_loss(rout_src, 'real')
        g_loss_cls =  self.classification_loss(rout_cls, label_org)
        # new_label = self.get_distance(label).detach()
        g_loss_rec = torch.mean(torch.abs(image - rec))
        # g_loss_rec = 0.6*torch.mean(torch.abs(image - rec)) + \
        #              0.4*torch.mean(torch.abs(image*new_label - rec*new_label))

        return cur_dices, cur_samples, g_loss_seg, g_loss_fake, g_loss_cls, g_loss_rec

    def unpair_KL_loss(self, con1, con2):
        # con1 = con1.softmax(dim=1)
        # con2 = con2.softmax(dim=1)
        # g_loss_kl = F.kl_div(con1.log(), con2, reduction='sum')
        # return (-g_loss_kl/10).exp()
        return torch.tensor(0.0)

    def train(self):
        loaders = {}
        loaders['train_pair'] = self.loaders['train_pair']
        # loaders['train_unpair'] = self.loaders['train_unpair']
        g_lr = self.g_lr
        d_lr = self.d_lr
        start_epoch = 0
        cur_step = -1
        if self.opt.use_weight:
            weight = torch.Tensor([0.1, 0.9])
        else:
            weight = None

        print('\nStart training...')
        start_time = time.time()
        if self.furthertrain:
            self.restore_model(self.furthertrain_epoch)
            start_epoch = self.furthertrain_epoch

        for epoch in range(start_epoch, self.max_epoch):
            self.G.train()
            if self.wtrans:
                self.D.train()

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #
            for p in loaders.keys():
                if p == 'train_pair':
                    flag_pair = True

                else:
                    flag_pair = False
                for i, batch_data in enumerate(loaders[p]):
                    cur_step += 2
                    # batch_data = [image1, label1, vec1, label_org1, name1
                    #                  0       1     2        3        4
                    #            , image2, label2, vec2, label_org2, name2]
                    #                 5      6      7         8       9
                    #           images1_c1, labels1_c1, images1_c2, labels1_c2, \
                    #                 10         11           12         13
                    #           images2_c1, labels2_c1, images2_c2, labels2_c2
                    #                 14         15          16          17

                    # image1 = batch_data[0].to(self.device)
                    # label1 = batch_data[1].to(self.device)
                    # label_org1 = batch_data[3].to(self.device)
                    # vec1 = batch_data[2].to(self.device)
                    # image1_change = batch_data[10].to(self.device)
                    # lable1_change = batch_data[12].to(self.device)
                    #
                    # image2 = batch_data[5].to(self.device)
                    # label2 = batch_data[6].to(self.device)
                    # label_org2 = batch_data[8].to(self.device)
                    # vec2 = batch_data[7].to(self.device)
                    # image2_change = batch_data[11].to(self.device)
                    # lable2_change = batch_data[13].to(self.device)

                    for train_type in range(0, 3):
                        if train_type != 2 and ((epoch + 1) >= self.pretrain_epoch):
                            continue
                        if train_type == 2 and ((epoch + 1) < self.pretrain_epoch):
                            continue

                        if train_type == 2 :
                            image1 = batch_data[0].to(self.device)
                            label1 = batch_data[1].to(self.device)
                            label_org1 = batch_data[3].to(self.device)
                            vec1 = batch_data[2].to(self.device)

                            image2 = batch_data[5].to(self.device)
                            label2 = batch_data[6].to(self.device)
                            label_org2 = batch_data[8].to(self.device)
                            vec2 = batch_data[7].to(self.device)
                        elif train_type == 0 :
                            image1 = batch_data[10].to(self.device)
                            label1 = batch_data[11].to(self.device)
                            label_org1 = batch_data[3].to(self.device)
                            vec1 = batch_data[2].to(self.device)

                            image2 = batch_data[12].to(self.device)
                            label2 = batch_data[13].to(self.device)
                            label_org2 = batch_data[3].to(self.device)
                            vec2 = batch_data[2].to(self.device)
                        elif train_type == 1:
                            image2 = batch_data[14].to(self.device)
                            label2 = batch_data[15].to(self.device)
                            label_org2 = batch_data[8].to(self.device)
                            vec2 = batch_data[7].to(self.device)

                            image1 = batch_data[16].to(self.device)
                            label1 = batch_data[17].to(self.device)
                            label_org1 = batch_data[8].to(self.device)
                            vec1 = batch_data[7].to(self.device)
                        else:
                            continue

                        loss = {}
                    # =================================================================================== #
                    #                             2. Train the discriminator                              #
                    # =================================================================================== #
                        if self.wtrans:
                            _, _, fake_image1, fake_image2, rec_x, rec_y, content1, content2, _, _= \
                                self.G(image1, vec1, image2, vec2)

                            # modality 1
                            d_loss_real1, d_loss_cls1, d_loss_fake1, d_loss_gp1 = self.trans_adv_loss(
                                image1, label_org1, rec_x, fake_image1, flag_pair)
                            d_loss_cadv1 = self.cadv_loss(content1, vec1, 'D')
                            d_loss_x = self.lambda_real * d_loss_real1 + self.lambda_fake * d_loss_fake1 + \
                                       self.lambda_cls * d_loss_cls1 + self.lambda_gp * d_loss_gp1 + \
                                       self.lambda_cadv * d_loss_cadv1
                            self.reset_grad()
                            d_loss_x.backward()
                            self.d_optimizer.step()


                            # modality 2
                            d_loss_real2, d_loss_cls2, d_loss_fake2, d_loss_gp2 = self.trans_adv_loss(
                                image2, label_org2, rec_y, fake_image2, flag_pair)
                            d_loss_cadv2 = self.cadv_loss(content2, vec2, 'D')
                            d_loss_y = self.lambda_real * d_loss_real2 + self.lambda_fake * d_loss_fake2 + \
                                       self.lambda_cls * d_loss_cls2 + self.lambda_gp * d_loss_gp2 + \
                                       self.lambda_cadv * d_loss_cadv2
                            self.reset_grad()
                            d_loss_y.backward()
                            self.d_optimizer.step()


                            d_loss_gp = d_loss_gp1 + d_loss_gp2
                            d_loss_real = d_loss_real1 + d_loss_real2
                            d_loss_cls = d_loss_cls1 + d_loss_cls2
                            d_loss_fake = d_loss_fake1 + d_loss_fake2
                            d_loss_cadv = d_loss_cadv1 + d_loss_cadv2

                            loss['D/r'] = d_loss_real.item()
                            loss['D/cls'] = d_loss_cls.item()
                            loss['D/cadv'] = d_loss_cadv.item()
                            loss['D/f'] = d_loss_fake.item()
                            loss['D/gp'] = d_loss_gp.item()

                        # =================================================================================== #
                        #                               3. Train the generator                                #
                        # =================================================================================== #
                        if (i + 1) % self.n_critic == 0:

                            #----------------------------------------
                            cur_dices = 0
                            cur_samples = 0
                            g_loss_seg = 0
                            g_loss_con = 0
                            g_loss_sty = 0
                            # for pair data
                            if flag_pair or train_type != 2:
                                # print(flag_pair)
                                #  modality1
                                seg_x1, seg_y1, fake_image1, fake_image2, rec_x, rec_y, content_x, content_y, \
                                style_x, style_y = self.G(image1, vec1, image2, vec2)

                                # seg_adv_loss1
                                g_loss_fake1, g_loss_cls1, g_loss_rec1, g_loss_trans1 = self.seg_adv_loss(
                                    image1, label1, fake_image1, rec_x, label_org1, image2, label_org2, train_type)
                                g_loss_cadv1 = self.cadv_loss(content_x, vec1, 'G')

                                g_loss_sty = self.sty_loss(style_x) + self.sty_loss(style_y)

                                # seg_loss1
                                cur_dices1, cur_samples1, g_loss_seg1, g_loss_con1 = self.seg_seg_loss(
                                    seg_x1, label1, content_x, content_y, weight)

                                if self.wtrans:
                                    g_loss_x = self.lambda_fake * g_loss_fake1 + self.lambda_cls * g_loss_cls1 + \
                                                self.lambda_rec * g_loss_rec1 + self.lambda_cadv * g_loss_cadv1 + \
                                                self.lambda_seg * g_loss_seg1
                                    if self.wlstran :
                                        g_loss_x += self.lambda_trans * g_loss_trans1
                                else:
                                    g_loss_x = self.lambda_seg * g_loss_seg1
                                if self.wlscon and train_type == 2:
                                    g_loss_x += self.lambda_con * g_loss_con1
                                    g_loss_con = g_loss_con1
                                if train_type != 2:
                                    g_loss_sty += torch.mean(torch.abs(style_x - style_y))
                                g_loss_x += self.lambda_sty * g_loss_sty

                                g_loss_seg = g_loss_seg1
                                cur_dices = cur_dices1
                                cur_samples = cur_samples1

                                # back1
                                self.reset_grad()
                                g_loss_x.backward()
                                self.g_optimizer.step()

                                #  modality2
                                seg_x1, seg_y1, fake_image1, fake_image2, rec_x, rec_y, content_x, content_y, \
                                style_x, style_y = self.G(image1, vec1, image2, vec2)

                                # seg_adv_loss2
                                g_loss_fake2, g_loss_cls2, g_loss_rec2, g_loss_trans2 = self.seg_adv_loss(
                                    image2, label2, fake_image2, rec_y, label_org2, image1, label_org1, train_type)
                                g_loss_cadv2 = self.cadv_loss(content_y, vec2, 'G')

                                g_loss_sty2 = self.sty_loss(style_x) + self.sty_loss(style_y)

                                cur_dices2, cur_samples2, g_loss_seg2, g_loss_con2 = self.seg_seg_loss(
                                    seg_y1, label2, content_y, content_x, weight)

                                if self.wtrans:
                                    g_loss_y = self.lambda_fake * g_loss_fake2 + self.lambda_cls * g_loss_cls2 + \
                                               self.lambda_rec * g_loss_rec2 + self.lambda_cadv * g_loss_cadv2 + \
                                               self.lambda_seg * g_loss_seg2
                                    if self.wlstran :
                                        g_loss_y += self.lambda_trans * g_loss_trans2
                                else:
                                    g_loss_y = self.lambda_seg * g_loss_seg2
                                if self.wlscon and train_type == 2:
                                    g_loss_y += self.lambda_con * g_loss_con2
                                    g_loss_con += g_loss_con2

                                if train_type != 2:
                                    g_loss_sty2 += torch.mean(torch.abs(style_x - style_y))

                                g_loss_y += self.lambda_sty * g_loss_sty2

                                g_loss_sty += g_loss_sty2
                                g_loss_seg += g_loss_seg2
                                cur_dices += cur_dices2
                                cur_samples += cur_samples2

                                # back2
                                self.reset_grad()
                                g_loss_y.backward()
                                self.g_optimizer.step()

                                # print
                                if self.wtrans:
                                    g_loss_fake = g_loss_fake1 + g_loss_fake2
                                    g_loss_cls = g_loss_cls1 + g_loss_cls2
                                    g_loss_cadv = g_loss_cadv1 + g_loss_cadv2
                                    g_loss_rec = g_loss_rec1 + g_loss_rec2


                                    loss['G/f'] = g_loss_fake.item()
                                    loss['G/cls'] = g_loss_cls.item()
                                    loss['G/cadv'] = g_loss_cadv.item()
                                    loss['G/r'] = g_loss_rec.item()
                                    loss['G/sty'] = g_loss_sty.item()

                                    loss['G/s'] = g_loss_seg.item()
                                    loss['dps'] = cur_dices / cur_samples

                                    if self.wlscon and train_type == 2:
                                        loss['G/con'] = g_loss_con.item()

                                    if self.wlstran:
                                        g_loss_trans = g_loss_trans1 + g_loss_trans2
                                        loss['G/t'] = g_loss_trans.item()

                            # for unpair data
                            else:

                                for m in range(2):
                                    seg_x1, seg_y1, fake_image1, fake_image2, rec_x, rec_y, content_x, content_y, \
                                    style_x, style_y = self.G(image1, vec1, image2, vec2 )
                                    g_loss_sty = self.sty_loss(style_x) + self.sty_loss(style_y)
                                    if m == 0:
                                        cur_dices, cur_samples, g_loss_seg, g_loss_fake, g_loss_cls, g_loss_rec = self.unpair_seg_loss(
                                            image1, seg_x1, rec_x, fake_image1, label1, label_org1, label_org2, weight)
                                        g_loss_cadv = self.cadv_loss(content_x, vec1, 'G')
                                        g_loss_kl = self.unpair_KL_loss(content_x, content_y)
                                    else:
                                        cur_dices, cur_samples, g_loss_seg, g_loss_fake, g_loss_cls, g_loss_rec = self.unpair_seg_loss(
                                            image2, seg_y1, rec_y, fake_image2, label2, label_org2, label_org1, weight)
                                        g_loss_cadv = self.cadv_loss(content_y, vec2, 'G')
                                        g_loss_kl = self.unpair_KL_loss(content_y, content_x)

                                    if self.wtrans:
                                        g_loss = self.lambda_seg * g_loss_seg + self.lambda_rec * g_loss_rec + \
                                                 self.lambda_fake * g_loss_fake + self.lambda_cls * g_loss_cls + \
                                                 self.lambda_cadv * g_loss_cadv + self.lambda_kl * g_loss_kl + \
                                                 self.lambda_sty * g_loss_sty

                                        loss['G/f'] = g_loss_fake.item()
                                        loss['G/cls'] = g_loss_cls.item()
                                        loss['G/cadv'] = g_loss_cadv.item()
                                        loss['G/r'] = g_loss_rec.item()
                                        loss['G/sty'] = g_loss_sty.item()
                                    else:
                                        g_loss = self.lambda_seg * g_loss_seg

                                    loss['G/s'] = g_loss_seg.item()
                                    loss['dps'] = cur_dices / cur_samples
                                    loss['G/kl'] = g_loss_kl.item()
                                    self.reset_grad()
                                    g_loss.backward()
                                    self.g_optimizer.step()

                        # =================================================================================== #
                        #                                 4. Miscellaneous                                    #
                        # =================================================================================== #
                        if (cur_step + 1) % self.log_step == 0:
                            et = time.time() - start_time
                            et = str(datetime.timedelta(seconds=et))[:-7]
                            line = "Elapsed [{}], Epoch [{}/{}], Iters [{}]".format(et, epoch + 1, self.max_epoch,
                                                                                    cur_step)
                            for k, v in loss.items():
                                if k == "G/sty":
                                    line += ", {}: {:.8f}".format(k, v)
                                else:
                                    line += ", {}: {:.4f}".format(k, v)
                                if self.use_tensorboard:
                                    # self.writer.add_scalar(k, v, (cur_step + 1) // self.n_critic)
                                    self.writer.add_scalar(k, v, (cur_step + 1))
                            print(line)

            if (epoch + 1) % self.val_epoch == 0:
                print()
                dps = self.val(epoch + 1)
                if self.use_tensorboard:
                    # self.writer.add_scalar('val/dps', dps, (epoch + 1) // self.n_critic)
                    self.writer.add_scalar('val/dps', dps, (epoch + 1))

            # Decay learning rates.
            if (epoch + 1) % self.lr_update_epoch == 0 and (epoch + 1) > (self.max_epoch - self.decay_epoch):
                g_dlr = self.g_lr - self.min_g_lr
                g_lr -= g_dlr / (self.decay_epoch / self.lr_update_epoch)
                d_dlr = self.d_lr - self.min_d_lr
                d_lr -= d_dlr / (self.decay_epoch / self.lr_update_epoch)
                self.update_lr(g_lr, d_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))
            if self.use_tensorboard:
                self.writer.add_scalar('G/g_lr', g_lr, epoch + 1)
                self.writer.add_scalar('D/d_lr', d_lr, epoch + 1)

    def val(self, epoch):
        self.G.eval()
        loaders = {}
        loaders['val'] = self.loaders['val']

        # Start training.
        print('Start Testing at iter {}...'.format(epoch))
        cur_samples = 0
        cur_dices = 0
        flag_pair = False

        with torch.no_grad():
            for p in loaders.keys():
                vis_index = []
                if p == 'val_pair':
                    flag_pair = True
                else:
                    flag_pair = False
                for k in range(4):
                    vis_index.append(random.randint(0, len(loaders[p]) - 1))
                for i, batch_data in enumerate(loaders[p]):
                    # batch_data = [image1, label1, vec1, label_org1, name1
                    #                  0       1     2        3        4
                    #            , image2, label2, vec2, label_org2, name2]
                    #                 5      6      7         8       9
                    image1 = batch_data[0]
                    label1 = batch_data[1]
                    vec1 = batch_data[2]
                    image1 = image1.to(self.device)
                    label1 = label1.to(self.device)
                    vec1 = vec1.to(self.device)
                    image1_c1 = batch_data[10].to(self.device)
                    label1_c1 = batch_data[11].to(self.device)
                    image1_c2 = batch_data[12].to(self.device)
                    label1_c2 = batch_data[13].to(self.device)
                    vis_list = [self.denorm(image1).cpu(), label1.unsqueeze(1).float().cpu()]
                    vis_list2 = [self.denorm(image1_c1).cpu(),label1_c1.unsqueeze(1).float().cpu(),
                                 self.denorm(image1_c2).cpu(),label1_c2.unsqueeze(1).float().cpu()]

                    image2 = batch_data[5]
                    label2 = batch_data[6]
                    vec2 = batch_data[7]
                    image2 = image2.to(self.device)
                    label2 = label2.to(self.device)
                    vec2 = vec2.to(self.device)
                    image2_c1 = batch_data[14].to(self.device)
                    label2_c1 = batch_data[15].to(self.device)
                    image2_c2 = batch_data[16].to(self.device)
                    label2_c2 = batch_data[17].to(self.device)
                    vis_list.append(self.denorm(image2).cpu())
                    vis_list.append(label2.unsqueeze(1).float().cpu())
                    vis_list2.append(self.denorm(image2_c1).cpu())
                    vis_list2.append(label2_c1.unsqueeze(1).float().cpu())
                    vis_list2.append(self.denorm(image2_c2).cpu())
                    vis_list2.append(label2_c2.unsqueeze(1).float().cpu())

                    seg_x1, seg_y1, fake_image1, fake_image2, rec_x, rec_y, contentx, contenty, _, _ = \
                        self.G(image1, vec1, image2, vec2)

                    # vis_list2 = contentx.split(1, dim=1).float().cpu()
                    # vis_list2 += contenty.split(1, dim=1).float().cpu()

                    _, pred1 = torch.max(seg_x1[0], 1)
                    _, pred2 = torch.max(seg_y1[0], 1)

                    vis_list.append(pred1.unsqueeze(1).float().cpu())
                    vis_list.append(pred2.unsqueeze(1).float().cpu())

                    cur_dices += dice_score(pred1, label1.detach(), 1, reduce=True) * label1.size(0)
                    cur_dices += dice_score(pred2, label2.detach(), 1, reduce=True) * label2.size(0)

                    cur_samples += label1.size(0)
                    cur_samples += label2.size(0)
                    if self.wtrans:
                        vis_list.append(self.denorm(fake_image1).cpu())
                        vis_list.append(self.denorm(fake_image2).cpu())

                        vis_list.append(self.denorm(rec_x).cpu())
                        vis_list.append(self.denorm(rec_y).cpu())

                    if i in vis_index:
                        vis_list = torch.cat(vis_list, dim=3)
                        sample_path = os.path.join(self.sample_dir, '{}-images-{}.jpg'.format(epoch, i))
                        save_image(vis_list.data.cpu(), sample_path, nrow=1, padding=0)

                        vis_list2 = torch.cat(vis_list2, dim=3)
                        sample_path2 = os.path.join(self.sample_dir, '{}-images_change-{}.jpg'.format(epoch, i))
                        save_image(vis_list2.data.cpu(), sample_path2, nrow=1, padding=0)

        # Log.
        dps = cur_dices / cur_samples
        if dps > self.best_dice or epoch == self.max_epoch or epoch == 200 or epoch%5 == 0:
            self.best_epoch = epoch
            self.best_dice = dps
            self.save_model(epoch)
        print_str = 'Cur dice per sample: {:.4f}. Best dice per sample: {:.4f} in Epoch {}.'.format(dps,
                                                                                                    self.best_dice,
                                                                                                    self.best_epoch)
        print(print_str)
        print()
        return dps

    def per_image(self, image, vec, name, alldic):
        compute_z = ['2']
        brain_dir = '/home/psdz/workplace/Aiyan/BRATS_TRAIN_2020'
        name_dic = {}
        sty = None
        syn_image, sty = self.G(image, vec, sty)

        value1 = [-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2]
        vlist = []
        base_image = []
        left_image = []
        right_image = []
        for i in range(len(sty[0])):
            simage = None
            if str(i) in compute_z:
                name_dic[str(i)] = {'t1ce':[], 't1':[], 't2':[], 'flair':[]}
            for j in range(len(value1)):
                sty2 = sty.clone()
                sty2[0][i] = value1[j]
                syn_image, _ = self.G(image, vec, sty2)
                if simage == None:
                    simage = syn_image
                else:
                    simage = torch.cat([simage, syn_image], dim=0)
                if j == 0:
                    left_image.append(syn_image)
                if j == int(len(value1)/2):
                    base_image.append(syn_image)
                if j == len(value1) - 1:
                    right_image.append(syn_image)
                if str(i) in compute_z:
                    compute_ssim(brain_dir, self.denorm(syn_image)*255.0, name,name_dic[str(i)])

            vlist.append(self.denorm(simage).cpu())
        alldic[name] = name_dic
        map = []
        for i in range(len(left_image)):
            map.append(self.denorm(self.denorm(left_image[i]) - self.denorm(base_image[i])).cpu())
        left_image = torch.cat(map, dim=3)
        # print(left_image.size())
        left_image = torch.squeeze(left_image, dim=0)
        left_image = torch.squeeze(left_image, dim=0)
        left_image = (left_image.detach().numpy() * 255).astype(np.uint8)

        left_image = cv2.applyColorMap(left_image, cv2.COLORMAP_HSV)
        cv2.imwrite(self.save_sty_image_map+'/lmap'+name, left_image)

        map = []
        for i in range(len(right_image)):
            map.append(self.denorm(self.denorm(right_image[i]) - self.denorm(base_image[i])).cpu())
        right_image = torch.cat(map, dim=3)
        # print(right_image.size())
        right_image = torch.squeeze(right_image, dim=0)
        right_image = torch.squeeze(right_image, dim=0)
        right_image = (right_image.detach().numpy() * 255).astype(np.uint8)

        right_image = cv2.applyColorMap(right_image, cv2.COLORMAP_HSV)
        cv2.imwrite(self.save_sty_image_map + '/rmap' + name, right_image)

        vlist = torch.cat(vlist, dim=3)
        save_image(vlist, self.save_sty_image+'/'+name, nrow=1, padding=0)
        return sty

    def infer(self, epoch, method='forward'):
        from PIL import Image


        save_dir = os.path.join(self.result_dir, str(epoch))
        check_dirs(save_dir)
        self.restore_model(epoch)
        self.G.eval()

        print('Start Testing at iter {}...'.format(epoch))
        print(method)
        with torch.no_grad():
            dic = {}
            alldic = {}
            for i, (image1, _, vec1, idx1, name1, image2, _, vec2, idx2, name2, _, _, _, _, _, _, _, _) in enumerate(self.loaders['test']):
                # batch_data = [image1, label1, vec1, label_org1, name1
                #                  0       1     2        3        4
                #            , image2, label2, vec2, label_org2, name2]
                #                 5      6      7         8       9

                image1 = image1.to(self.device)
                vec1 = vec1.to(self.device)
                idx1 = idx1.to(self.device)

                image2 = image2.to(self.device)
                vec2 = vec2.to(self.device)
                idx2 = idx2.to(self.device)


                if method == 'styablation':

                    for b in range(image1.size()[0]):
                        for i in range(2):
                            if i==0:
                                image = torch.unsqueeze(image1[b, ...], dim=0)
                                vec = torch.unsqueeze(vec1[b, ...], dim=0)
                                name = name1[b]
                            else:
                                image = torch.unsqueeze(image2[b, ...], dim=0)
                                vec = torch.unsqueeze(vec2[b, ...], dim=0)
                                name = name2[b]
                            print(name)
                            dic[name] = self.per_image(image, vec, name, alldic).cpu().numpy().tolist()
                    continue

                if method == 'forward':
                    seg_x1, seg_y1, fake_image1, fake_image2, rec_x, rec_y, _, _, _, _ = \
                        self.G(image1, vec1, image2, vec2 )
                    # seg_x1, seg_y1, fake_image1, fake_image2, rec_x, rec_y, _, _
                    _, preds1 = torch.max(seg_x1[0], 1)
                    _, preds2 = torch.max(seg_y1[0], 1)

                else:
                    exit('Unknown Methods!')

                preds1 = preds1.cpu().numpy()
                preds2 = preds2.cpu().numpy()
                # fake_image1 = fake_image1.cpu().numpy()
                # fake_image2 = fake_image2.cpu().numpy()
                # rec_x = rec_x.cpu().numpy()
                # rec_y = rec_y.cpu().numpy()
                for b in range(preds1.shape[0]):
                    modal, pid, index, pn, _ = parse_image_name(name1[b])
                    if self.save_rf == False:
                        pred = preds1[b, ...]
                        pred[pred == 1] = 255
                        img = Image.fromarray(pred.astype('uint8'))
                        check_dirs(os.path.join(save_dir, '{}'.format(modal)))
                        img.save(os.path.join(save_dir, '{}/{}_{}.png'.format(modal, pid, index)))
                    else:
                        # print(fake_image1.shape)
                        fi1 = fake_image1[b, ...]
                        #print('fil shape')

                        # fi1 = np.squeeze(fi1,axis=0)
                        # fi1 = Image.fromarray(fi1.astype('uint8'))


                        rc = rec_x[b, ...]
                        # rc = np.squeeze(rc,axis=0)
                        # rc = Image.fromarray(rc.astype('uint8'))

                        fi1_path = os.path.join(self.generation_dir,
                                   '{}/{}_{}_{}_{}.png'.format(self.selected_modal[idx1[b]],self.selected_modal[idx2[b]], pid, index,pn))
                        rc_path = os.path.join(self.generation_dir,
                                   '{}/{}_{}_{}_{}.png'.format(self.selected_modal[idx1[b]],self.selected_modal[idx1[b]], pid, index,pn))

                        # fi1.save(fi1_path)
                        # rc.save(rc_path)
                        save_image(self.denorm(fi1).cpu(), fi1_path, nrow=1, padding=0)
                        save_image(self.denorm(rc).cpu(), rc_path, nrow=1, padding=0)

                for b in range(preds2.shape[0]):
                    modal, pid, index, pn, _ = parse_image_name(name2[b])
                    if self.save_rf == False:
                        pred = preds2[b, ...]
                        pred[pred == 1] = 255
                        img = Image.fromarray(pred.astype('uint8'))
                        check_dirs(os.path.join(save_dir, '{}'.format(modal)))
                        img.save(os.path.join(save_dir, '{}/{}_{}.png'.format(modal, pid, index)))
                    else:
                        fi1 = fake_image2[b, ...]
                        # fi1 = np.squeeze(fi1, axis=0)
                        # fi1 = Image.fromarray(fi1.astype('uint8'))
                        rc = rec_y[b, ...]
                        # rc = np.squeeze(rc, axis=0)
                        # rc = Image.fromarray(rc.astype('uint8'))
                        fi2_path = os.path.join(self.generation_dir,
                                    '{}/{}_{}_{}_{}.png'.format(self.selected_modal[idx2[b]],self.selected_modal[idx1[b]], pid, index,pn))
                        rc_path = os.path.join(self.generation_dir,
                                    '{}/{}_{}_{}_{}.png'.format(self.selected_modal[idx2[b]],self.selected_modal[idx2[b]], pid, index,pn))

                        # fi1.save(fi1_path)
                        # rc.save(rc_path)
                        save_image(self.denorm(fi1).cpu(), fi2_path, nrow=1, padding=0)
                        save_image(self.denorm(rc).cpu(), rc_path, nrow=1, padding=0)

            with open(self.sty_json_dir, 'w') as f:
                json.dump(dic, f)

            with open(self.ssim_json_dir, 'w') as f:
                json.dump(alldic, f)

        return


if __name__ == '__main__':
    cudnn.benchmark = True

    args = argparse.ArgumentParser()
    args.add_argument('--train_pair_list', type=str)
    args.add_argument('--train_unpair_list', type=str)
    args.add_argument('--val_list', type=str)
    args.add_argument('--test_list', type=str)

    # Data Loader.
    args.add_argument('--phase', type=str, default='train')
    args.add_argument('--selected_modal', nargs='+', default=['t2', 'flair'])
    args.add_argument('--image_size', type=int, default=128)
    args.add_argument('--batch_size', type=int, default=8)
    args.add_argument('--num_workers', type=int, default=4)
    args.add_argument('--seg_type', type=str, default='WT')
    # Model configurations.
    args.add_argument('--out_channels', type=int, default=2)
    args.add_argument('--content_channel', type=int, default=4)
    args.add_argument('--feature_maps', type=int, default=64)
    args.add_argument('--levels', type=int, default=4)
    args.add_argument('--nz', type=int, default=8)
    args.add_argument('--norm_type', type=str, default='instance')
    args.add_argument('--use_dropout', type=bool, default=True)
    args.add_argument('--wskip', type=bool, default=True)
    args.add_argument('--wtrans', type=bool, default=True)
    args.add_argument('--wlscon', type=bool, default=True)
    args.add_argument('--wlstran', type=bool, default=True)
    args.add_argument('--save_rf', type=bool, default=False)
    args.add_argument('--d_conv_dim', type=int, default=64)
    args.add_argument('--d_repeat_num', type=int, default=6)

    # Lambda.
    args.add_argument('--lambda_cls', type=float, default=1)
    args.add_argument('--lambda_rec', type=float, default=50)
    args.add_argument('--lambda_gp', type=float, default=1)
    args.add_argument('--lambda_seg', type=float, default=100)
    args.add_argument('--lambda_con', type=float, default=10)
    args.add_argument('--lambda_shape', type=float, default=10)
    args.add_argument('--lambda_trans', type=float, default=100)
    args.add_argument('--lambda_real', type=float, default=1)
    args.add_argument('--lambda_fake', type=float, default=1)
    args.add_argument('--lambda_cadv', type=float, default=10)
    args.add_argument('--lambda_kl', type=float, default=1)
    args.add_argument('--lambda_sty', type=float, default=10)

    # Train configurations.
    args.add_argument('--max_epoch', type=int, default=50)
    args.add_argument('--decay_epoch', type=int, default=10)
    args.add_argument('--pretrain_epoch', type=int, default=20)
    args.add_argument('--furthertrain_epoch', type=int, default=60)
    args.add_argument('--furthertrain', type=bool, default=False)
    args.add_argument('--g_lr', type=float, default=1e-4)
    args.add_argument('--min_g_lr', type=float, default=1e-6)
    args.add_argument('--d_lr', type=float, default=1e-4)
    args.add_argument('--min_d_lr', type=float, default=1e-6)
    args.add_argument('--beta1', type=float, default=0.9)
    args.add_argument('--beta2', type=float, default=0.999)
    args.add_argument('--ignore_index', type=int, default=None)
    args.add_argument('--seg_loss_type', type=str, default='cross-entropy')
    args.add_argument('--seed', type=int, default=1234)
    args.add_argument('--use_weight', type=bool, default=True)
    args.add_argument('--n_critic', type=int, default=2)

    # Test configurations.
    args.add_argument('--test_epoch', nargs='+', default=None)
    args.add_argument('--method', type=str, default='forward')

    # Miscellaneous.
    args.add_argument('--use_tensorboard', type=bool, default=True)
    args.add_argument('--device', type=bool, default=True)
    args.add_argument('--gpu_id', type=str, default='0')

    # Directories.
    args.add_argument('--checkpoint_dir', type=str)

    # Step size.
    args.add_argument('--log_step', type=int, default=40)
    args.add_argument('--val_epoch', type=int, default=5)
    args.add_argument('--lr_update_epoch', type=int, default=1)
    args = args.parse_args()

    print('-----Config-----')
    for k, v in sorted(vars(args).items()):
        print('%s:\t%s' % (str(k), str(v)))
    print('-------End------\n')

    # Set Random Seed.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Set GPU ID
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    data_files = dict(train_pair=args.train_pair_list, train_unpair=args.train_unpair_list,
                      val=args.val_list, test=args.test_list)

    solver = Solver(data_files, args)
    if args.phase == 'train':
        solver.train()
    elif args.phase == 'test':
        for test_iter in args.test_epoch:
            test_iter = int(test_iter)
            solver.infer(test_iter, args.method)
            print()

    print('Done!')





