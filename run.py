import sys

import cv2
import numpy as np
import torch
import torch.optim as optim

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as mpanim
from matplotlib.gridspec import GridSpec

from pathlib import Path

from baseline_3d_pose.utils import *
from baseline_3d_pose.model import *
from baseline_3d_pose.viz import *

indexx = int(sys.argv[1])

class Options():
    def __init__(self):
        # paths
        self.data_path = Path('data')
        self.model_path = Path('model')

        # train options
        self.actions = 'All'
        self.attempt_id = '01'
        self.attempt_path = Path('model')/self.attempt_id

        self.load_ckpt = False

        # train hyper-params
        self.bs = 128
        self.epochs = 10
        self.lr = 1e-3

        # model hyper-params
        self.size = 1024
        self.stages = 2
        self.dropout = 0.5


stat_3d = torch.load(data_path/'stat_3d.pt')
stat_2d = torch.load(data_path/'stat_2d.pt')
train_set_2d = torch.load(data_path/'train_2d.pt')
rcams = torch.load(data_path/'rcams.pt')
mean_2d = stat_2d['mean']
std_2d = stat_2d['std']
dim_use_2d = stat_2d['dim_use']
dim_ignore_2d = stat_2d['dim_ignore']
mean_3d = stat_3d['mean']
std_3d = stat_3d['std']
dim_use_3d = stat_3d['dim_use']
dim_ignore_3d = stat_3d['dim_ignore']

imgs_path = Path("imgs")
json_path = Path("json")
data_path = Path('data')

img_seq = imgs_path/'crop_img_seq/'
out_seq = imgs_path/'crop_out_seq/'
kp_seq = json_path/'crop_json/'

options = torch.load('model/01/options.pt')

model = Model()
model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=options.lr)

model_state = torch.load(options.attempt_path/'last_model.pt')
optimizer_state = torch.load(options.attempt_path/'last_optimizer.pt')
model.load_state_dict(model_state)
optimizer.load_state_dict(optimizer_state)

model.eval()

img_lists = sorted(img_seq.ls())
out_lists = sorted(out_seq.ls())
kp_lists = sorted(kp_seq.ls())

idx = indexx
# idx = 167
img = mpimg.imread(img_lists[idx])
out = mpimg.imread(out_lists[idx])
kp = coco_to_skel(get_kp_from_json(kp_lists[idx]) * 2)

kp_norm = normalize_kp(kp, mean_2d, std_2d, dim_use_2d)
kp_unnorm = unnormalize_data(kp_norm, mean_2d, std_2d, dim_ignore_2d)

key = list(train_set_2d.keys())[4]
key

kp_t = torch.from_numpy(kp_norm).float()
kp_t3d = model(kp_t.cuda())
kp_3d = kp_t3d.cpu().detach().numpy()
kp_3d = unnormalize_data(kp_3d, mean_3d, std_3d, dim_ignore_3d)
kp_3d = cam_to_world_centered(kp_3d, key, rcams)

# %matplotlib qt
fig = plt.figure(figsize=(13,13))
gs = GridSpec(2, 2)

ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])
ax3 = plt.subplot(gs[3], projection='3d')
ax3.view_init(elev=20, azim=70)

ax0.set_title('Input Image')
ax1.set_title('COCO Pose')
ax2.set_title('Custom Pose')
ax3.set_title('3D Pose Predictions')

ax0.imshow(img)
ax1.imshow(out)

show_2d_pose(kp_unnorm, ax2)
ax2.invert_yaxis()

show_3d_pose(kp_3d, ax3)

plt.tight_layout()
plt.show()