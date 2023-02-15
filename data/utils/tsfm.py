# -*- coding: utf-8 -*-
import numpy as np
import torch
import PIL
from scipy import ndimage
from skimage.transform import resize
from skimage.util import random_noise
import random
# import elasticdeform
import numbers
import torchvision.transforms.functional as F



# Transforms for PIL.Image
class RandomCrop:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h-th)
        j = random.randint(0, w-tw)
        return i, j, th, tw

    def __call__(self, img1, lbl1, img2, lbl2):
        i, j, h, w = self.get_params(img1, self.size)
        return F.crop(img1, i, j, h, w), F.crop(lbl1, i, j, h, w), F.crop(img2, i, j, h, w), F.crop(lbl2, i, j, h, w)


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img1, lbl1, img2, lbl2):
        if random.random() < self.p:
            return F.hflip(img1), F.hflip(lbl1), F.hflip(img2), F.hflip(lbl2)
        return img1, lbl1, img2, lbl2


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img1, lbl1, img2, lbl2):
        if random.random() < self.p:
            return F.vflip(img1), F.vflip(lbl1), F.vflip(img2), F.vflip(lbl2)
        return img1, lbl1, img2, lbl2


class Rotate:
    def __init__(self, degrees):
        self.degrees = (-degrees, degrees)

    @staticmethod
    def get_params(degrees):
        angle = random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, img1, lbl1, img2, lbl2):
        angle = self.get_params(self.degrees)
        img1 = F.rotate(img1, angle, interpolation = F.InterpolationMode.BICUBIC)
        img2 = F.rotate(img2, angle, interpolation = F.InterpolationMode.BICUBIC)
        lbl1 = F.rotate(lbl1, angle, interpolation = F.InterpolationMode.NEAREST)
        lbl2 = F.rotate(lbl2, angle, interpolation = F.InterpolationMode.NEAREST)
        return img1, lbl1, img2, lbl2


class Zoom:
    def __init__(self, low, high, min_size):
        self.scales = (low, high)
        self.min_size = min_size

    @staticmethod
    def get_params(scales):
        scale = random.uniform(scales[0], scales[1])
        return scale

    def __call__(self, img1, lbl1, img2, lbl2):
        scale = self.get_params(self.scales)
        size = int(scale * img1.size[0])
        if size < self.min_size:
            size = self.min_size
        img1 = F.resize(img1, size, interpolation = F.InterpolationMode.BICUBIC)
        img2 = F.resize(img2, size, interpolation = F.InterpolationMode.BICUBIC)
        lbl1 = F.resize(lbl1, size, interpolation = F.InterpolationMode.NEAREST)
        lbl2 = F.resize(lbl2, size, interpolation = F.InterpolationMode.NEAREST)
        return img1, lbl1, img2, lbl2



class ToLongTensor:
    def __init__(self, type):
        self.type = type
    def __call__(self, lbl):
        lbl = np.array(lbl, dtype='uint8')
        if self.type == 'ET':
            lbl[lbl == 120] = 0  # 0 60 120 180 240
            lbl[lbl == 60] = 0
            lbl[lbl == 180] = 0
            # lbl[lbl != 240] = 0
        elif self.type == 'TC':  #
            lbl[lbl == 120] = 0
        elif self.type != 'WT':
            print('******************************************error type!!')
        lbl[lbl != 0] = 1
        return torch.from_numpy(lbl).long()


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img1, lbl1, img2, lbl2):
        for t in self.transforms:
            img1, lbl1, img2, lbl2 = t(img1, lbl1, img2, lbl2)
        return img1, lbl1, img2, lbl2

