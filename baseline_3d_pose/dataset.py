# AUTOGENERATED! DO NOT EDIT! File to edit: 02_dataset.ipynb (unless otherwise specified).

__all__ = ['Human36Dataset']

# Cell
import os
import torch
from torch.utils.data import Dataset, DataLoader
from .utils import *
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Cell
class Human36Dataset(Dataset):
    def __init__(self, actions, data_path, is_train=True):
        self.actions, self.data_path, self.is_train = actions, data_path, is_train
        self.inp_list, self.out_list, self.key_list = [], [], []

        if self.is_train:
            self.data_2d = torch.load(data_path/'train_2d.pt')
            self.data_3d = torch.load(data_path/'train_3d.pt')
        else:
            self.data_2d = torch.load(data_path/'test_2d.pt')
            self.data_3d = torch.load(data_path/'test_3d.pt')

        for key in self.data_2d.keys():
            assert self.data_2d[key].shape[0] == self.data_3d[key].shape[0]
            if key[1] in self.actions:
                num_file = self.data_2d[key].shape[0]
                for i in range(num_file):
                    self.inp_list.append(self.data_2d[key][i])
                    self.out_list.append(self.data_3d[key][i])
                    self.key_list.append(key)

    def __getitem__(self, idx):
        inp = torch.from_numpy(self.inp_list[idx]).float()
        out = torch.from_numpy(self.out_list[idx]).float()
        return inp, out

    def get_key(self, idx):
        return self.key_list[idx]

    def __len__(self):
        return len(self.inp_list)