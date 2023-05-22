import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

T_max = 3
dt = 0.02

MAX_TRAJ_STEPS = int(T_max//dt) # 150
EMBED_DIM = 16 # (x, xdot, xddot, o, odot, cup_type) x=(x, y, z), o=(phi, thetha, psi) cup_type = 1 or 2 or 3
DATA_NUM = 2


def get_initials(data_num):
    if data_num == 0:
        x, y, z = 0.2, -1, 0.1
        xdot, ydot, zdot = 0, 0, 0
        xddot, yddot, zddot = 0, 0.4, 0.3
        y_train = 1 # slosh free
    else:
        x, y, z = 0.2, -1, 0.1
        xdot, ydot, zdot = 0, 0, 0
        xddot, yddot, zddot = 1, 3, 10
        y_train = 0 # slosh
    return y_train, x, y, z, xdot, ydot, zdot, xddot, yddot, zddot


def generate_fake_data():
    # Rough estimate of Panda's reachable area: x: (0, 1.57), y: (-1, 1), z: (0, 1)
    x_train = np.zeros((DATA_NUM, MAX_TRAJ_STEPS, EMBED_DIM), dtype=np.float64)
    y_train = np.zeros((DATA_NUM, 1))
    
    phi, theta, psi = 0, 0, 0
    phidot, thetadot, psidot = 0, 0, 0
    cup_type = 1

    for d in range(DATA_NUM):
        y_train[d], x, y, z, xdot, ydot, zdot, xddot, yddot, zddot = get_initials(d)
        zddot_slow_down = 3.7*zddot/(MAX_TRAJ_STEPS//2)
        
        for i in range(MAX_TRAJ_STEPS):
            x_train[d, i, :] = np.array([x, xdot, xddot, y, ydot, yddot, z, zdot, zddot, phi, phidot, theta, thetadot, psi, psidot, cup_type])

            x += xdot * dt
            y += ydot * dt
            z += zdot * dt
            xdot += xddot * dt
            ydot += yddot * dt
            zdot += zddot * dt
            
            if i >= MAX_TRAJ_STEPS//4 and i <= 3 * MAX_TRAJ_STEPS//4:
                zddot -= zddot_slow_down


    return x_train, y_train


def visualize_data(x_train, y_train):
    # for d in range(DATA_NUM):
    d = 0
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([0, 1.57])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1])
    ax.scatter(x_train[d, :, 0], x_train[d, :, 3], x_train[d, :, 6])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # plt.show()
    plt.savefig('data_safe_trajectory.png')

# x_train, y_train = generate_fake_data()
# visualize_data(x_train, y_train)


def get_data(debug=False):
    if debug:
        # x_train = np.random.uniform(5, size=(DATA_NUM, MAX_TRAJ_STEPS, EMBED_DIM))
        # y_train = np.random.randint(1, size=(DATA_NUM, 1, 1))
        # x_val = np.random.uniform(5, size=(DATA_NUM, MAX_TRAJ_STEPS, EMBED_DIM))
        # y_val = np.random.randint(1, size=(DATA_NUM, 1, 1))
        x_train, y_train = generate_fake_data()
        x_val, y_val = x_train, y_train
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
