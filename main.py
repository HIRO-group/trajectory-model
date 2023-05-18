import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, \
    Layer, Input, Embedding, GlobalAveragePooling1D, Dense, Layer

'''
T max = 10 second
dt = 0.02 second
max trajectory lenght = 10/0.02 = 500
number of features per step = 15 (x, xdot, xddot, o, odot) x=(x, y, z), o=(phi, thetha, psi)
'''
MAX_TRAJECTORY_LENGHT = 500
TRAJECTORY_FEATURES_NUM = 15


def get_angles(pos, k, d):
    i = k // 2
    angles = pos/(np.power(10000,2*i/d))
    return angles

def positional_encoding(positions, d):
    angle_rads = get_angles(np.arange(positions)[:, np.newaxis], # col
                            np.arange(d)[np.newaxis, :], # row
                            d)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


x = np.random.randint(5, size=(MAX_TRAJECTORY_LENGHT, TRAJECTORY_FEATURES_NUM))

pos_encoding = positional_encoding(MAX_TRAJECTORY_LENGHT, TRAJECTORY_FEATURES_NUM)
x = x + pos_encoding[:, :, :]