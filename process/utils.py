import os
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
from PIL import Image
from skimage import measure

def CRFs(image, label):
    label = np.array(label)
    label[label != 0] = 1
    image = np.array(image)

    n_labels = len(set(label.flat))

    use_2d = True

    if use_2d:

        d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], n_labels)
        U = unary_from_labels(label, n_labels, gt_prob=0.9, zero_unsure=False)
        d.setUnaryEnergy(U)

        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

    else:

        d = dcrf.DenseCRF(image.shape[1] * image.shape[0], n_labels)

        U = unary_from_labels(label, n_labels, gt_prob=0.9, zero_unsure=False)
        d.setUnaryEnergy(U)

        feats = create_pairwise_gaussian(sdims=(3, 3), shape=image.shape[:2])
        d.addPairwiseEnergy(feats, compat=8, kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(10)

    MAP = np.argmax(Q, axis=0)
    MAP = np.uint8(MAP)
    MAP[MAP != 0] = 1
    MAP = MAP.reshape((128, 128))

    return MAP

def connected_components(pre,seg_type):
    labels = measure.label(pre, connectivity=2)
    retain_num = []
    ratio = 0.2
    thresold = ratio * np.sum(labels!=0)
    for j in range(1, np.max(labels)+1):
        if np.sum(labels==j)> thresold:
            retain_num.append(j)
        else:
            labels[labels==j] = 0
    labels[labels!=0] = 1

    return np.uint8(labels)

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return


def pad_zero(s, length=3):
    s = str(s)
    assert len(s) < length + 1
    if len(s) < length:
        s = '0' * (length - len(s)) + s
    return s

def zscore(x):
    x = (x - x.mean()) / x.std()
    return x


def percentile(x, prct):
    low, high = np.percentile(x, prct[0]), np.percentile(x, prct[1])
    x[x < low] = low
    x[x > high] = high
    return x


def parse_image_name(name):
    #  T2_001_023_0.png
    n = name.split('.')[0]
    mod, pid, index, pn = n.split('_')
    return mod, pid, index, pn, 'modality'+name[len(mod):]

def parse_image_name2(name, modal_num):
    #  T2_001_023_0.png
    n = name.split('.')[0]
    mod, pid, index, pn = n.split('_')
    return mod, pid, index, pn, '%'+str(modal_num)+name[len(mod):]

def center_crop(img, size):
    h, w = img.shape
    x, y = (h - size) // 2, (w - size) // 2
    img_ = img[x: x+size, y: y+size]
    return img_
