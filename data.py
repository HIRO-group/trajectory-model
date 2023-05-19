import tensorflow as tf
import numpy as np

'''
T max = 10 second
dt = 0.02 second
max trajectory lenght = 10/0.02 = 500
number of features per step = 15 (x, xdot, xddot, o, odot) x=(x, y, z), o=(phi, thetha, psi)
'''

MAX_TRAJ_STEPS = 500
EMBED_DIM = 15
DATA_NUM = 3

def get_data(debug=False):
    if debug:
        x_train = np.random.uniform(5, size=(DATA_NUM, MAX_TRAJ_STEPS, EMBED_DIM))
        y_train = np.random.randint(1, size=(DATA_NUM, 1, 1))
        x_val = np.random.uniform(5, size=(DATA_NUM, MAX_TRAJ_STEPS, EMBED_DIM))
        y_val = np.random.randint(1, size=(DATA_NUM, 1, 1))
        return x_train, y_train, x_val, y_val
    else:
        #have to make sure that the data is between a certain range so positional encoding works properly
        raise NotImplementedError


def get_angles(pos, k, d):
    i = k // 2
    angles = pos/(np.power(10000,2*i/d))
    return angles


def positional_encoding(positions, d): 
    angle_rads = get_angles(np.arange(positions)[:, np.newaxis],
                            np.arange(d)[np.newaxis, :],
                            d)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32) # (1, MAX_TRAJ_STEPS, TRAJ_FEATURES_NUM)


def process_data(x_train, x_val):
    pos_encoding = positional_encoding(MAX_TRAJ_STEPS, EMBED_DIM)
    x_train = x_train + pos_encoding[:, :, :]
    x_val = x_val + pos_encoding[:, :, :]
    return x_train, x_val
