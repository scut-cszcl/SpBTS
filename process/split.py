
import os
import random
import argparse
import numpy as np
from utils import check_dir, parse_image_name
from PIL import Image


def split(brain_dir, save_dir, n_unpair_train=20, n_pair_train=4, n_unpair_val=60, n_pair_val=30, n_split=3, exclude_threshold=400, upsample=True):

    patient2site = {}
    patient_ids = np.zeros(shape=(0))
    patient_ids = np.array([f for f in os.listdir(brain_dir)])
    print('all patient_ids:',patient_ids)
    perm = np.random.permutation(len(patient_ids))
    patient_ids = patient_ids[perm]

    n_test = len(patient_ids) - n_unpair_train - n_pair_train - n_unpair_val - n_pair_val

    splits = []
    i = 0
    for ns in range(n_split):
        splits.append(dict(val_unpair=patient_ids[i: i + n_unpair_val],
                           val_pair=patient_ids[i + n_unpair_val: i + n_unpair_val + n_pair_val],
                           test=patient_ids[i+ n_unpair_val + n_pair_val: i + n_unpair_val + n_pair_val+n_test],
                           train_pair=patient_ids[i + n_unpair_val + n_pair_val+n_test: i + n_unpair_val + n_pair_val+n_test+n_pair_train],
                           train_unpair=np.concatenate([patient_ids[:i], patient_ids[i + n_unpair_val + n_pair_val+n_test+n_pair_train:]])))
        i += n_unpair_val + n_pair_val+n_test+n_pair_train
        if i + n_unpair_val + n_pair_val+n_test+n_pair_train > len(patient_ids):
            perm = np.random.permutation(len(patient_ids))
            patient_ids = patient_ids[perm]
            i = 0

    check_dir(save_dir)
    for i in range(n_split):
        print('len of train_unpair, train_pair, val_unpair, val_pair, and test:',len(splits[i]['train_unpair']),len(splits[i]['train_pair']),
              len(splits[i]['val_unpair']), len(splits[i]['val_pair']), len(splits[i]['test']))
        print(i,'split patient_train_unpair:',list(sorted(splits[i]['train_unpair'])))
        print(i,'split patient_train_pair:',list(sorted(splits[i]['train_pair'])))
        print(i,'split patient_val_unpair:',list(sorted(splits[i]['val_unpair'])))
        print(i,'split patient_val_pair:',list(sorted(splits[i]['val_pair'])))
        print(i,'split patient_test:',list(sorted(splits[i]['test'])))

        cls_rate = [0] * 5

        train_pair_f = os.path.join(save_dir, '{}-train_pair.txt'.format(i))
        train_unpair_f = os.path.join(save_dir, '{}-train_unpair.txt'.format(i))
        val_pair_f = os.path.join(save_dir, '{}-val_pair.txt'.format(i))
        val_unpair_f = os.path.join(save_dir, '{}-val_unpair.txt'.format(i))
        test_f = os.path.join(save_dir, '{}-test.txt'.format(i))
        train_pair_f = open(train_pair_f, 'w')
        train_unpair_f = open(train_unpair_f, 'w')
        val_pair_f = open(val_pair_f, 'w')
        val_unpair_f = open(val_unpair_f, 'w')
        test_f = open(test_f, 'w')

        writer = dict(train_pair=train_pair_f, train_unpair=train_unpair_f, val_pair=val_pair_f, val_unpair=val_unpair_f, test=test_f)
        lines = dict(train_pair=[[], []], train_unpair=[[], []], val_pair=[], val_unpair=[], test=[])
        for phase in ('train_pair', 'train_unpair', 'val_pair', 'val_unpair', 'test'):
            for pid in sorted(splits[i][phase]):
                target_dir = os.path.join(brain_dir, pid)
                image_dir = os.path.join(target_dir,'flair')
                label_dir = os.path.join(target_dir,'Label')
                for f in sorted(os.listdir(image_dir)):
                    m, pid, index, pn, na = parse_image_name(f)


                    fpath = os.path.join(target_dir, 'modality')
                    fpath = os.path.join(fpath, na)
                    lpath = os.path.join(label_dir, '{}_{}.png'.format(pid, index))

                    fpath2 = os.path.join(image_dir, f)

                    lbl = np.array(Image.open(lpath))
                    image = np.array(Image.open(fpath2))
                    if np.sum(image != 0) < exclude_threshold:
                        continue


                    cmap = [0, 60, 120, 180, 240]
                    for c, cm in enumerate(cmap):
                        t = np.sum(lbl == cm)
                        cls_rate[c] += t

                    line = '{} {}\n'.format(fpath, lpath)
                    line = line.replace('\\','/')
                    if phase == 'train_unpair' or phase == 'train_pair':
                        lines[phase][int(pn)].append(line)
                    else:
                        lines[phase].append(line)

        if upsample:
            for p in ['train_unpair','train_pair']:
                n_negative, n_positive = len(lines[p][0]), len(lines[p][1])
                print('Before upsample ', p, n_negative, n_positive)
                for j in range(n_negative - n_positive):
                    rand_index = random.randint(0, n_positive - 1)
                    lines[p][1].append(lines[p][1][rand_index])
                n_negative, n_positive = len(lines[p][0]), len(lines[p][1])
                print('After upsample ', p, n_negative, n_positive)

        for phase in ('train_unpair', 'train_pair','val_pair' ,'val_unpair', 'test'):
            if phase == 'train_unpair' or phase == 'train_pair':
                tar_list = lines[phase][0] + lines[phase][1]
                tar_list = list(sorted(tar_list))
            else:
                tar_list = lines[phase]
            for line in tar_list:
                writer[phase].write(line)

        cls_rate = [cr / sum(cls_rate) for cr in cls_rate]
        print("cls_rate:")

        print(cls_rate)



if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--brain_dir', type=str, default='./dataset/BRATS_TRAIN_2020')
    parse.add_argument('--save_dir', type=str, default='../process/partition')
    parse.add_argument('--train_rate', type=int, default=4)
    parse.add_argument('--n_split', type=int, default=3)
    parse.add_argument('--mo', type=str, default='t1ce')
    parse.add_argument('--n_max', type=int, default=100, help='Maximum number of dataset.')
    parse.add_argument('--seed', type=int, default=1234)
    parse.add_argument('--exclude_nolabel', type=int, default=1600,
                       help='Exclude images when #label smaller than threshold.')
    parse.add_argument('--upsample', type=bool, default=True, help='Balance the unlabel images.')
    opt = parse.parse_args()

    random.seed(opt.seed)
    np.random.seed(opt.seed)

    split(opt.brain_dir, opt.save_dir, n_split=opt.n_split,
          exclude_threshold=opt.exclude_nolabel, upsample=opt.upsample)
    print('Done!')
