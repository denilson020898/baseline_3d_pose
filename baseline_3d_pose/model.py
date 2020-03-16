# AUTOGENERATED! DO NOT EDIT! File to edit: 03_model.ipynb (unless otherwise specified).

__all__ = ['init_kaiming', 'ResLinear', 'Model']

# Cell
import torch
import torch.nn as nn
import torch.nn.functional as F

# Cell
def init_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)

# Cell
class ResLinear(nn.Module):
    def __init__(self, size, pd=0.5):
        super().__init__()
        self.size = size
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(pd)
        # learnable
        self.ln1 = nn.Linear(self.size, self.size)
        self.bn2 = nn.BatchNorm1d(self.size)
        self.ln3 = nn.Linear(self.size, self.size)
        self.bn4 = nn.BatchNorm1d(self.size)
    def forward(self, x):
        y = self.drop(self.relu(self.bn2(self.ln1(x))))
        y = self.drop(self.relu(self.bn4(self.ln3(y))))
        return x + y

# Cell
class Model(nn.Module):
    def __init__(self, size=1024, num_res_lyr=2, pd=0.5):
        super().__init__()
        self.size, self.num_res_lyr, self.pd = size, num_res_lyr, pd
        self.input_size, self.output_size = 32, 48
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(self.pd)

        # input size
        self.ln_in = nn.Linear(self.input_size, self.size)
        self.bn_in = nn.BatchNorm1d(self.size)

        # res layers
        self.lins = []
        for i in range(num_res_lyr):
            self.lins.append(ResLinear(self.size, self.pd))
        self.lins = nn.ModuleList(self.lins)

        # output size
        self.ln_out = nn.Linear(self.size, self.output_size)
    def forward(self, x):
        y = self.drop(self.relu(self.bn_in(self.ln_in(x))))
        for i in range(self.num_res_lyr):
            y = self.lins[i](y)
        y = self.ln_out(y)
        return y