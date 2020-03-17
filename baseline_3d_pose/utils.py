# AUTOGENERATED! DO NOT EDIT! File to edit: 00_utils.ipynb (unless otherwise specified).

__all__ = ['TRAIN_SUBJECTS', 'TEST_SUBJECTS', 'H36M_NAMES', 'N_CAMERAS', 'N_JOINTS', 'PLOT_RADIUS', 'data_path',
           'get_actions', 'normalize_data', 'unnormalize_data', 'normalize_kp', 'get_kp_from_json', 'coco_to_skel',
           'get_cam_rt', 'camera_to_world_frame', 'cam_to_world_centered', 'show_2d_pose', 'show_3d_pose']

# Cell
import json
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from fastai.vision import *

# Cell
TRAIN_SUBJECTS = [1, 5, 6, 7, 8]
TEST_SUBJECTS  = [9, 11]

H36M_NAMES = ['']*32
H36M_NAMES[0]  = 'Hip'
H36M_NAMES[1]  = 'RHip'
H36M_NAMES[2]  = 'RKnee'
H36M_NAMES[3]  = 'RFoot'
H36M_NAMES[6]  = 'LHip'
H36M_NAMES[7]  = 'LKnee'
H36M_NAMES[8]  = 'LFoot'
H36M_NAMES[12] = 'Spine'
H36M_NAMES[13] = 'Thorax'
H36M_NAMES[14] = 'Neck/Nose'
H36M_NAMES[15] = 'Head'
H36M_NAMES[17] = 'LShoulder'
H36M_NAMES[18] = 'LElbow'
H36M_NAMES[19] = 'LWrist'
H36M_NAMES[25] = 'RShoulder'
H36M_NAMES[26] = 'RElbow'
H36M_NAMES[27] = 'RWrist'

N_CAMERAS = 4
N_JOINTS = 32

PLOT_RADIUS = 300

# Cell
data_path = Path('data')
data_path.ls()

# Cell
def get_actions(action):
    """
    """
    actions = ['Directions',
               'Discussion',
               'Eating',
               'Greeting',
               'Phoning',
               'Photo',
               'Posing',
               'Purchases',
               'Sitting',
               'SittingDown',
               'Smoking',
               'Waiting',
               'WalkDog',
               'Walking',
               'WalkTogether']
    if action == 'All' or action == 'all':
        return actions

    if action not in actions:
        raise (ValueError, f'{action} is not found in {x for x in actions}')

    return [action]

# Cell
def normalize_data(unnormalized, mean, std, dim_use):
    normalized = {}
    for key in unnormalized.keys():
        unnormalized[key] = unnormalized[key][:, dim_use]
        m = mean[dim_use]
        s = std[dim_use]
        normalized[key] = np.divide((unnormalized[key] - m), s)
    return normalized

# Cell
def unnormalize_data(normalized, mean, std, dim_ignore):
    T = normalized.shape[0]
    D = mean.shape[0]
    orig = np.zeros((T, D), dtype=np.float32)
    dim_use = np.array([dim for dim in range(D) if dim not in dim_ignore])
    orig[:, dim_use] = normalized

    std_m = std.reshape((1, D))
    std_m = np.repeat(std_m, T, axis=0)
    mean_m = mean.reshape((1, D))
    mean_m = np.repeat(mean_m, T, axis=0)
    orig = np.multiply(orig, std_m) + mean_m
    return orig

# Cell
def normalize_kp(kp, mean, std, dim_use):
    m = mean[dim_use]
    s = std[dim_use]
    return np.divide((kp - m), s)

# Cell
def get_kp_from_json(fname):
    with open(fname) as f:
        kp = json.load(f)
    kpl = np.array(kp['people'][0]['pose_keypoints_2d'])
    return kpl

# Cell
def coco_to_skel(s):
    s = s.reshape(-1, 2)
    hip = (s[8] + s[11]) / 2
    rhip = s[8]
    rknee = s[9]
    rfoot = s[10]
    lhip = s[11]
    lknee = s[12]
    lfoot = s[13]
    spine = (s[1] + hip) / 2
    thorax = s[1]
    head = (s[16] + s[17]) / 2 # TODO:  kurang tinggi
    lshoulder = s[5]
    lelbow = s[6]
    lwrist = s[7]
    rshoulder = s[2]
    relbow = s[3]
    rwrist = s[4]
    return np.array([hip, rhip, rknee, rfoot, lhip, lknee, lfoot,
            spine, thorax, head,
            lshoulder, lelbow, lwrist,
            rshoulder, relbow, rwrist ]).reshape(1, -1)

# Cell
def get_cam_rt(key, rcams):
    subj, _, sname = key
    cname = sname.split('.')[1] # <-- camera name
    scams = {(subj,c+1): rcams[(subj,c+1)] for c in range(N_CAMERAS)} # cams of this subject
    scam_idx = [scams[(subj,c+1)][-1] for c in range(N_CAMERAS)].index( cname ) # index of camera used
    the_cam  = scams[(subj, scam_idx+1)] # <-- the camera used
    R, T, f, c, k, p, name = the_cam
    assert name == cname
    return R, T

# Cell
def camera_to_world_frame(P, R, T):
  X_cam = R.T.dot( P.T ) + T
  return X_cam.T

# Cell
def cam_to_world_centered(data, key, rcams):
    R, T = get_cam_rt(key, rcams)
    data_3d_worldframe = camera_to_world_frame(data.reshape((-1, 3)), R, T)
    data_3d_worldframe = data_3d_worldframe.reshape((-1, N_JOINTS*3))
    # subtract root translation
    return data_3d_worldframe - np.tile( data_3d_worldframe[:,:3], (1,N_JOINTS) )

# Cell
def show_2d_pose(skel, ax, lcolor='#094e94', rcolor='#940909'):
    kps = np.reshape(skel, (len(H36M_NAMES), -1))
    start = np.array([1,2,3,1,7,8,1, 13,14,14,18,19,14,26,27])-1 # start points
    end = np.array([2,3,4,7,8,9,13,14,16,18,19,20,26,27,28])-1 # end points
    left_right = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    for i in range(len(start)):
        x, y = [np.array( [kps[start[i], j], kps[end[i], j]] ) for j in range(2)]
        ax.plot(x, y, lw=2, c=lcolor if left_right[i] else rcolor)
        ax.scatter(x, y, c=lcolor if left_right[i] else rcolor)
    xroot, yroot = kps[0,0], kps[0,1]
    ax.set_xlim(-PLOT_RADIUS+xroot, PLOT_RADIUS+xroot)
    ax.set_ylim(-PLOT_RADIUS+yroot, PLOT_RADIUS+yroot)

# Cell
def show_3d_pose(skel, ax, lcolor='#094e94', rcolor='#940909'):
    kps = np.reshape(skel, (len(H36M_NAMES), -1))
    start = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1 # start points
    end = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1 # end points
    left_right = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    for i in np.arange( len(start) ):
        x, y, z = [np.array( [kps[start[i], j], kps[end[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=2, c=lcolor if left_right[i] else rcolor)
        ax.scatter(x, y, z, c=lcolor if left_right[i] else rcolor)

    xroot, yroot, zroot = kps[0,0], kps[0,1], kps[0,2]
    ax.set_xlim3d([-PLOT_RADIUS*2+xroot, PLOT_RADIUS*2+xroot])
    ax.set_zlim3d([-PLOT_RADIUS*2+zroot, PLOT_RADIUS*2+zroot])
    ax.set_ylim3d([-PLOT_RADIUS*2+yroot, PLOT_RADIUS*2+yroot])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    white = (1.0, 1.0, 1.0, 0.0)
    ax.w_xaxis.set_pane_color(white)
    ax.w_yaxis.set_pane_color(white)
    ax.w_xaxis.line.set_color(white)
    ax.w_yaxis.line.set_color(white)
    ax.w_zaxis.line.set_color(white)