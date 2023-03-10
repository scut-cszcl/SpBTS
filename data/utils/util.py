import os
import torch
import torch.nn as nn

def print_net(model):
    print(model)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
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
