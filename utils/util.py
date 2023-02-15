import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

def parse_image_name(name):

    n = name.split('.')[0]
    mod, pid, index, pn = n.split('_')
    return mod, pid, index, pn, 'modality'+name[len(mod):]

def print_net(model):
    print(model)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)  #
    n = n / 1000000
    print('[*] has {:.4f}M parameters!'.format(n))


def check_dirs(path):
    if type(path) not in (tuple, list):
        path = [path]
    for p in path:
        if not os.path.exists(p):
            os.makedirs(p)
    return


def dice_score(preds, masks, cls, eps=1e-10, reduce=False):
    """ preds, masks is Tensor liked BxHxW
    """
    b = masks.size(0)
    pred, mask = (preds == cls), (masks == cls)
    inter = (pred & mask)

    inter = inter.sum(2).sum(1).float()
    pred = pred.sum(2).sum(1).float()
    mask = mask.sum(2).sum(1).float()

    dices = 2 * inter / (pred + mask + eps)

    if reduce:
        dices = dices.sum() / b
    return  dices

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    # channel = 1
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def compute_ssim(brain_dir, simage, name, dic):

    _, pid, index, pn, _ = parse_image_name(name)
    image_t1ce = '{}/{}/{}/{}_{}_{}_{}.png'.format(brain_dir, str(int(pid)), 't1ce', 't1ce',pid,index,pn)
    image_t1 = '{}/{}/{}/{}_{}_{}_{}.png'.format(brain_dir, str(int(pid)), 't1', 't1',pid,index,pn)
    image_t2 = '{}/{}/{}/{}_{}_{}_{}.png'.format(brain_dir, str(int(pid)), 't2', 't2',pid,index,pn)
    image_flair = '{}/{}/{}/{}_{}_{}_{}.png'.format(brain_dir, str(int(pid)), 'flair', 'flair',pid,index,pn)

    s_image = simage.to(torch.float64).to(torch.device('cuda'))

    target_image_t1ce = np.array(Image.open(image_t1ce))
    s_target_image_t1ce = torch.from_numpy(target_image_t1ce).unsqueeze(0).unsqueeze(0).to(torch.float64).to(torch.device('cuda'))

    target_image_t1 = np.array(Image.open(image_t1))
    s_target_image_t1 = torch.from_numpy(target_image_t1).unsqueeze(0).unsqueeze(0).to(torch.float64).to(torch.device('cuda'))

    target_image_t2 = np.array(Image.open(image_t2))
    s_target_image_t2 = torch.from_numpy(target_image_t2).unsqueeze(0).unsqueeze(0).to(torch.float64).to(torch.device('cuda'))

    target_image_flair = np.array(Image.open(image_flair))
    s_target_image_flair = torch.from_numpy(target_image_flair).unsqueeze(0).unsqueeze(0).to(torch.float64).to(torch.device('cuda'))

    dic['t1ce'].append(float(ssim(s_image, s_target_image_t1ce)))
    dic['t1'].append(float(ssim(s_image, s_target_image_t1)))
    dic['t2'].append(float(ssim(s_image, s_target_image_t2)))
    dic['flair'].append(float(ssim(s_image, s_target_image_flair)))

def pre_process(seg_x1, seg_y1, seg_type):
    vaule = 0.6
    pred1_0, pred1_1, pred2_0, pred2_1 = 0, 0, 0, 0
    if seg_type in ['TE', 'TandE']:
        pred1_0 = torch.sigmoid(seg_x1[0][:, 0, ...])
        pred1_1 = torch.sigmoid(seg_x1[0][:, 1, ...])
        pred2_0 = torch.sigmoid(seg_y1[0][:, 0, ...])
        pred2_1 = torch.sigmoid(seg_y1[0][:, 1, ...])
    elif seg_type in ['WT', 'TC', 'ET', 'NCR']:
        seg_x1[0] = seg_x1[0].softmax(dim=1)
        seg_y1[0] = seg_y1[0].softmax(dim=1)
        pred1_0 = seg_x1[0][:, 0, :, :]
        pred1_1 = seg_x1[0][:, 1, :, :]
        pred2_0 = seg_y1[0][:, 0, :, :]
        pred2_1 = seg_y1[0][:, 1, :, :]
    else:
        print('Error seg_type !!!!')

    pred1_0[pred1_0 > vaule] = 1
    pred1_0[pred1_0 != 1] = 0
    pred1_1[pred1_1 > vaule] = 1
    pred1_1[pred1_1 != 1] = 0
    pred2_0[pred2_0 > vaule] = 1
    pred2_0[pred2_0 != 1] = 0
    pred2_1[pred2_1 > vaule] = 1
    pred2_1[pred2_1 != 1] = 0

    return pred1_0, pred1_1, pred2_0, pred2_1


if __name__ == '__main__':
    a = torch.Tensor([[[0, 0, 2], [1, 0, 3]], [[0, 0, 2], [1, 0, 3]]])
    b = torch.Tensor([[[1, 0, 3], [1, 0, 3]], [[0, 0, 2], [1, 0, 3]]])
    a = dice_score(a, b, 0, reduce=True)
    print(a * b.size(0))
