import torch


##=============================================================##
##====================  dice_loss  ===========================##
##=============================================================##
def d_l(outputs, targets):
    targets = targets.float()
    eps = 1e-7
    inse = (outputs * targets).sum().float()
    l = (outputs * outputs).sum().float()
    r = targets.sum().float()
    dice = 2.0 * inse / (l + r + eps)
    return 1.0 - 1.0 * dice

def d_loss(outputs, targets):
    loss = 0.0
    for b in range(outputs.size(0)):
        loss += d_l(outputs[b,...], targets[b,...])
    return loss / outputs.size(0)

def dice_loss(seg, label, seg_type):
    loss = 0.0
    loss2 = 0.0
    if seg_type == 'TE':
        label_1 = label.clone()
        label_1[label_1 != 1] = 0
        label_1and4 = label.clone()
        label_1and4[label_1and4 == 4] = 1
        label_1and4[label_1and4 != 1] = 0
        loss = d_loss(torch.sigmoid(seg[:,0,...]), label_1and4)
        loss2 = d_loss(torch.sigmoid(seg[:,1,...]), label_1)
    elif seg_type == 'TandE':
        label_1 = label.clone()
        label_1[label_1 != 4] = 0
        label_1[label_1 != 0] = 1
        label_1and4 = label.clone()
        label_1and4[label_1and4 == 4] = 1
        label_1and4[label_1and4 != 1] = 0
        loss = d_loss(torch.sigmoid(seg[:,0,...]), label_1and4)
        loss2 = d_loss(torch.sigmoid(seg[:,1,...]), label_1)
    elif seg_type == 'WT':
        target = label.clone()
        target[target != 0] = 1
        loss = d_loss(seg.softmax(dim=1)[:, 1, :, :], target)
    elif seg_type == 'TC':
        target = label.clone()
        target[target == 4] = 1
        target[target != 1] = 0
        loss = d_loss(seg.softmax(dim=1)[:, 1, :, :], target)
    elif seg_type == 'ET':
        target = label.clone()
        target[target != 4] = 0
        target[target != 0] = 1
        loss = d_loss(seg.softmax(dim=1)[:, 1, :, :], target)
    elif seg_type == 'NCR':
        target = label.clone()
        target[target != 1] = 0
        loss = d_loss(seg.softmax(dim=1)[:, 1, :, :], target)
    else:
        print('Error seg_type !!!!!!!')
    return [loss, loss2]



##=============================================================##
##====================  dice_score  ===========================##
##=============================================================##

def d_s(outputs, targets):
    outputs[outputs > 0.5] = 1
    outputs[outputs != 1] = 0
    targets = targets.float()
    eps = 1e-7
    inse = (outputs * targets).sum().float()
    l = (outputs * outputs).sum().float()
    r = targets.sum().float()
    dice = 2.0 * inse / (l + r + eps)
    return dice

def d_score(outputs, targets):
    dice = 0.0
    for b in range(outputs.size(0)):
        dice += d_s(outputs[b,...], targets[b,...])
    return dice

def dice_score(seg, label, seg_type):
    Dice = 0.0
    Dice2 = 0.0
    if seg_type == 'TE':
        label_1 = label.clone()
        label_1[label_1 != 1] = 0
        label_1and4 = label.clone()
        label_1and4[label_1and4 == 4] = 1
        label_1and4[label_1and4 != 1] = 0
        Dice = d_score(torch.sigmoid(seg[:,0,...]), label_1and4)
        Dice2 = d_score(torch.sigmoid(seg[:,1,...]), label_1)
    elif seg_type == 'TandE':
        label_1 = label.clone()
        label_1[label_1 != 4] = 0
        label_1[label_1 != 0] = 1
        label_1and4 = label.clone()
        label_1and4[label_1and4 == 4] = 1
        label_1and4[label_1and4 != 1] = 0
        Dice = d_score(torch.sigmoid(seg[:,0,...]), label_1and4)
        Dice2 = d_score(torch.sigmoid(seg[:,1,...]), label_1)
    elif seg_type == 'WT':
        target = label.clone()
        target[target != 0] = 1
        Dice = d_score(seg.softmax(dim=1)[:, 1, :, :], target)
    elif seg_type == 'TC':
        target = label.clone()
        target[target == 4] = 1
        target[target != 1] = 0
        Dice = d_score(seg.softmax(dim=1)[:, 1, :, :], target)
    elif seg_type == 'ET':
        target = label.clone()
        target[target != 4] = 0
        target[target != 0] = 1
        Dice = d_score(seg.softmax(dim=1)[:, 1, :, :], target)
    elif seg_type == 'NCR':
        target = label.clone()
        target[target != 1] = 0
        Dice = d_score(seg.softmax(dim=1)[:, 1, :, :], target)
    else:
        print('Error seg_type !!!!!!!')
    return [Dice, Dice2]
