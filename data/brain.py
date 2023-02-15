# -*- coding: utf-8 -*-
from torch.utils import data
from torchvision import transforms as T
from PIL import Image
import torch
import os
import random
import numpy as np
from utils import util, tsfm
from process.utils import parse_image_name
from elasticdeform import deform_random_grid

class Brain(data.Dataset):
    def __init__(self, data_file, selected_modal, inputs_transform=None, train_transform=None,
                 labels_transform=None, join_transform=None, phase='train_pair'):
        self.selected_modal = selected_modal
        self.c_dim = len(self.selected_modal)
        self.inputs_transform = inputs_transform
        self.labels_transform = labels_transform
        self.join_transform = join_transform
        self.train_transform = train_transform
        self.data_file = data_file
        self.modal2idx = {}
        self.idx2modal = {}
        self.dataset = {}
        self.container = []
        self.phase = phase
        self.init()
        self.get_container()

    def init(self):
        for i, modal_name in enumerate(self.selected_modal):
            self.modal2idx[modal_name] = i
            self.idx2modal[i] = modal_name
            self.dataset[modal_name] = []
        self.dataset['data'] = []
        lines = [line.rstrip() for line in open(self.data_file, 'r')]
        flag_m = 0
        flag_pid = "-1"
        for i, line in enumerate(lines):
            image, label = line.split()
            image_name = os.path.split(image)[1]
            _, pid, index, _, _ = parse_image_name(image_name)
            if self.phase == 'train_pair':
                self.dataset['data'].append([image, label, image_name])
            elif self.phase == 'train_unpair':
                if pid != flag_pid:
                    flag_pid = pid
                    flag_m += 1
                    flag_m %= len(self.selected_modal)
                image = image.replace('modality', self.selected_modal[flag_m])
                image_name = image_name.replace('modality', self.selected_modal[flag_m])
                self.dataset[self.selected_modal[flag_m]].append([image, label, image_name, flag_m])
            else:
                for i in range(len(self.selected_modal)):
                    modal = self.selected_modal[i]
                    ima = image.replace('modality', modal)
                    ima_name = image_name.replace('modality', modal)
                    self.dataset[modal].append([ima, label, ima_name, i])

        if self.phase == 'train_pair':
            print('[*] Load {}, which contains {} paired images and labels, {}'.format(self.data_file,
                                                                                       len(self.dataset['data']),
                                                                                       self.modal2idx))
        else:
            print('[*] Load {}, which contains:'.format(self.data_file))
            for m in self.selected_modal:
                print('{}-{} images and labels, {}'.format(m, len(self.dataset[m]), self.modal2idx))

    def get_container(self):
        list = [[0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 1, 2]]
        if self.phase == 'train_pair':
            for i in range(len(self.dataset['data'])):
                image, label, image_name = self.dataset['data'][i]
                n = i % len(list)

                image0 = image.replace('modality', self.selected_modal[list[n][0]])
                image1 = image.replace('modality', self.selected_modal[list[n][1]])
                image2 = image.replace('modality', self.selected_modal[list[n][2]])
                image3 = image.replace('modality', self.selected_modal[list[n][3]])

                image_name0 = image_name.replace('modality', self.selected_modal[list[n][0]])
                image_name1 = image_name.replace('modality', self.selected_modal[list[n][1]])
                image_name2 = image_name.replace('modality', self.selected_modal[list[n][2]])
                image_name3 = image_name.replace('modality', self.selected_modal[list[n][3]])

                self.container.append([[image0, label, image_name0, list[n][0]],
                                       [image1, label, image_name1, list[n][1]]])
                self.container.append([[image2, label, image_name2, list[n][2]],
                                       [image3, label, image_name3, list[n][3]]])
        else:
            for m in self.selected_modal:
                random.shuffle(self.dataset[m])
            self.length = []
            max = 0
            for m in self.selected_modal:
                max = len(self.dataset[m]) if len(self.dataset[m]) > max else max
                self.length.append(len(self.dataset[m]))
            for i in range(max):
                n = i % len(list)
                i0 = i if i < self.length[list[n][0]] else random.randint(0, self.length[list[n][0]] - 1)
                i1 = i if i < self.length[list[n][1]] else random.randint(0, self.length[list[n][1]] - 1)
                i2 = i if i < self.length[list[n][2]] else random.randint(0, self.length[list[n][2]] - 1)
                i3 = i if i < self.length[list[n][3]] else random.randint(0, self.length[list[n][3]] - 1)
                self.container.append([self.dataset[self.selected_modal[list[n][0]]][i0],
                                       self.dataset[self.selected_modal[list[n][1]]][i1]])
                self.container.append([self.dataset[self.selected_modal[list[n][2]]][i2],
                                       self.dataset[self.selected_modal[list[n][3]]][i3]])

    def __getitem__(self, idex):

        image1, label1, name1, idx1 = self.container[idex][0]
        image2, label2, name2, idx2 = self.container[idex][1]

        images1, labels1 = Image.open(image1), Image.open(label1)
        images2, labels2 = Image.open(image2), Image.open(label2)

        # images1_change = block_shuffle(images1)
        # images2_change = block_shuffle(images2)
        i1,i2,i3,i4 = random.randint(0, 100),random.randint(0, 100),random.randint(0, 100),random.randint(0, 100)
        i1,i2,i3,i4 = i1 % len(self.train_transform), i2 % len(self.train_transform),i3 % len(self.train_transform),i4 % len(self.train_transform)

        images1_c1, labels1_c1,_,_ = self.train_transform[i1](images1, labels1,images1, labels1)
        images1_c2, labels1_c2,_,_ = self.train_transform[i2](images1, labels1,images1, labels1)

        images2_c1, labels2_c1,_,_ = self.train_transform[i3](images2, labels2,images2, labels2)
        images2_c2, labels2_c2,_,_ = self.train_transform[i4](images2, labels2,images2, labels2)

        if self.join_transform:
            images1, labels1, images2, labels2 = self.join_transform(images1, labels1, images2, labels2)
        if self.inputs_transform:
            images1 = self.inputs_transform(images1)
            images2 = self.inputs_transform(images2)
            images1_c1 = self.inputs_transform(images1_c1)
            images1_c2 = self.inputs_transform(images1_c2)
            images2_c1 = self.inputs_transform(images2_c1)
            images2_c2 = self.inputs_transform(images2_c2)
        if self.labels_transform:
            labels1 = self.labels_transform(labels1)
            labels2 = self.labels_transform(labels2)
            labels1_c1 = self.labels_transform(labels1_c1)
            labels1_c2 = self.labels_transform(labels1_c2)
            labels2_c1 = self.labels_transform(labels2_c1)
            labels2_c2 = self.labels_transform(labels2_c2)

        vec1 = [0] * self.c_dim
        vec1[idx1] = 1

        vec2 = [0] * self.c_dim
        vec2[idx2] = 1

        return images1, labels1, torch.Tensor(vec1), idx1, name1, \
               images2, labels2, torch.Tensor(vec2), idx2, name2, \
               images1_c1, labels1_c1, images1_c2, labels1_c2, \
               images2_c1, labels2_c1, images2_c2, labels2_c2

    def __len__(self):
        if self.phase == 'train_pair' or self.phase == 'val_pair':
            return len(self.dataset['data'])
        else:
            return len(self.container)

# def ElasticDeform(img):
#     s = random.uniform(9,13)
#     img = np.array(img)
#     img = deform_random_grid(img, sigma=s, points=3, order=0, axis=1)
#     img = Image.fromarray(img)
#     return img

def ElasticDeform(img, msk):
    s = random.uniform(9, 13)
    img = np.array(img) / 255.; msk = np.array(msk)
    img, msk = deform_random_grid([img, msk], sigma=s, points=3, order=[3, 0])
    img *= 255
    img = Image.fromarray(img.astype(np.uint8)); msk = Image.fromarray(msk.astype(np.uint8))
    return img, msk

def block_shuffle(input):
    input = np.asarray(input.convert('L'))
    list = []
    for i in range(4):
        list.append(i)
    random.shuffle(list)
    output = np.zeros_like(input)
    for i in range(4):
        h = i // 2 * 64
        w = i % 2 * 64
        H = list[i] // 2 * 64
        W = list[i] % 2 * 64
        output[h:h + 64, w:w + 64] = input[H:H + 64, W:W + 64]
    return Image.fromarray(output)


def get_loaders(data_files, selected_modals, batch_size=16, num_workers=1, image_size=128, seg_type='WT'):
    train_join_tsfm = tsfm.Compose([
        tsfm.RandomHorizontalFlip(0.5),
        tsfm.RandomVerticalFlip(0.5),
        tsfm.Rotate(180),
        # tsfm.Zoom(0.9, 1.1, image_size),
        # tsfm.RandomCrop(image_size)
    ])
    train_tsfm = [tsfm.RandomHorizontalFlip(1),
                   tsfm.RandomVerticalFlip(1),
                   tsfm.Rotate(180),
                   tsfm.RandomScale(0.8, 1.2),
                   tsfm.ElasticDeform(4, 8),
                   tsfm.Randomtranslate(20, 20)]
    input_tsfm = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.5,), std=(0.5,))
    ])
    label_tsfm = T.Compose([
        tsfm.ToLongTensor(seg_type)
    ])

    datasets = dict(train_pair=Brain(data_files['train_pair'], selected_modals, inputs_transform=input_tsfm,
                        train_transform=train_tsfm, labels_transform=label_tsfm, join_transform=train_join_tsfm, phase='train_pair'),
                    train_unpair=Brain(data_files['train_unpair'], selected_modals, inputs_transform=input_tsfm,
                        train_transform=train_tsfm, labels_transform=label_tsfm, join_transform=train_join_tsfm, phase='train_unpair'),
                    val=Brain(data_files['val'], selected_modals, inputs_transform=input_tsfm,
                        train_transform=train_tsfm, labels_transform=label_tsfm, join_transform=None, phase='val'),
                    test=Brain(data_files['test'], selected_modals, inputs_transform=input_tsfm,
                        train_transform=train_tsfm, labels_transform=label_tsfm, join_transform=None, phase='test')
                    )
    loaders = {x: data.DataLoader(dataset=datasets[x], batch_size=batch_size,
                                  shuffle=(x == 'train_unpair' or x == 'train_pair'),
                                  num_workers=num_workers)
               for x in ('train_pair', 'train_unpair', 'val', 'test')}
    return loaders

