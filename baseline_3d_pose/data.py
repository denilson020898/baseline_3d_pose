# AUTOGENERATED! DO NOT EDIT! File to edit: 00_data.ipynb (unless otherwise specified).

__all__ = ['TRAIN_SUBJECTS', 'TEST_SUBJECTS', 'H36M_NAMES', 'get_actions', 'normalize', 'unnormalize']

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
def normalize(unnormalized, mean, std, dim_use):
    normalized = {}
    for key in unnormalized.keys():
        unnormalized[key] = unnormalized[key][:, dim_use]
        m = mean[dim_use]
        s = std[dim_use]
        normalized[key] = np.divide((unnormalized[key] - m), s)
    return normalized

# Cell
def unnormalize(normalized, mean, std, dim_ignore):
    """
    """
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