# -*- coding: utf-8 -*-
from torch.utils import data
from collections import OrderedDict
import os

# from process.utils import parse_image_name

def parse_image_name(name):
    #  T2_001_023_0.png
    n = name.split('.')[0]
    mod, pid, index, pn = n.split('_')
    return mod, pid, index, pn, 'modality'+name[len(mod):]

class Patient(data.Dataset):
    def __init__(self, data_file, select_modal):

        self.data_file = data_file
        self.select_modal = select_modal
        self.patients = dict()
        for modal in select_modal:
            self.patients[modal] = OrderedDict()
        self.keys = []
        self.init()

    def init(self):
        info = dict()
        lines = [line.rstrip() for line in open(self.data_file, 'r')]

        for i, line in enumerate(lines):
            image, label = line.split()
            image_name = os.path.split(image)[1]
            _, pid, index, _, _ = parse_image_name(image_name)

            for modal in self.select_modal:
                ima = image.replace('modality', modal)

                if modal not in self.select_modal:
                    print('***{} not in select_modal:'.format(modal))
                    continue
                if pid in self.patients[modal].keys():
                    self.patients[modal][pid].append((ima, label))
                else:
                    self.patients[modal][pid] = [(ima, label)]
                    self.keys.append([modal, pid])
                    if modal in info.keys():
                        info[modal] += 1
                    else:
                        info[modal] = 1
        print('Load {} patient volumes!'.format(len(self.keys)))
        print(info)

    def __getitem__(self, idx):
        modal = self.keys[idx][0]
        pid = self.keys[idx][1]
        return self.patients[modal][pid]

    def __len__(self):
        return len(self.keys)


