import numpy as np
import matplotlib.pyplot as plt

from process_data import process_data

def get_fake_initials(data_num):
    if data_num == 0:
        x, y, z = 0.2, -1, 0.1
        xdot, ydot, zdot = 0, 0, 0
        xddot, yddot, zddot = 0, 0.4, 0.3
        y_train = 0 # slosh free
    else:
        x, y, z = 0.2, -1, 0.1
        xdot, ydot, zdot = 0, 0, 0
        xddot, yddot, zddot = 1, 3, 10
        y_train = 1 # slosh
    return y_train, x, y, z, xdot, ydot, zdot, xddot, yddot, zddot


def generate_fake_data(data_num, max_traj_steps, embed_dim, dt):
    # Rough estimate of Panda's reachable area: x: (0, 1.57), y: (-1, 1), z: (0, 1)
    x_train = np.zeros((data_num, max_traj_steps, embed_dim), dtype=np.float64)
    y_train = np.zeros((data_num, 1))
    
    phi, theta, psi = 0, 0, 0
    phidot, thetadot, psidot = 0, 0, 0
    cup_type = 1

    for d in range(data_num):
        y_train[d], x, y, z, xdot, ydot, zdot, xddot, yddot, zddot = get_fake_initials(d)
        zddot_slow_down = 3.7*zddot/(max_traj_steps//2)
        
        for i in range(max_traj_steps):
            x_train[d, i, :] = np.array([x, xdot, xddot, y, ydot, yddot, z, zdot, zddot, phi, phidot, theta, thetadot, psi, psidot, cup_type])

            x += xdot * dt
            y += ydot * dt
            z += zdot * dt
            xdot += xddot * dt
            ydot += yddot * dt
            zdot += zddot * dt
            
            if i >= max_traj_steps//4 and i <= 3 * max_traj_steps//4:
                zddot -= zddot_slow_down
    return x_train, y_train


def get_data(data_num, max_traj_steps, embed_dim, dt, debug=False):
    if debug:
        x_train, y_train = generate_fake_data(data_num, max_traj_steps, embed_dim, dt)
        x_val, y_val = x_train, y_train
        return x_train, y_train, x_val, y_val, max_traj_steps
    else:
        X, Y = process_data()
        max_traj_steps = X.shape[1]
        train_id = int(3*X.shape[0]/4)
        train_index = np.random.choice(X.shape[0], train_id, replace=False) 
        val_index = np.setdiff1d(np.arange(X.shape[0]), train_index)
        X_train, Y_train = X[train_index], Y[train_index]
        X_val, Y_val = X[val_index], Y[val_index]
        return X_train, Y_train, X_val, Y_val, max_traj_steps, X, Y


def read_panda_data():
    file_address_prefix = '/home/ava/npm/trajectory-model/panda_data/'
    safe_data_addr = ['1_safe.npz', '2_safe.npz']
    unsafe_data_addr = ['3_unsafe.npz', '4_unsafe.npz', '5_safe.npz']
    
    num_data = len(safe_data_addr) + len(unsafe_data_addr)

    X = np.zeros((num_data, 2000, 8)) 
    Y = np.zeros((num_data, 1))

    for data_idx, addr in enumerate(safe_data_addr):
        npz_file = np.load(file_address_prefix + addr)
        positions_and_orientations = npz_file['positions_and_orientations']
        for idx, p_o in enumerate(positions_and_orientations[:2000]):
            X[data_idx, idx, 0:7] = p_o
            X[data_idx, idx, 7] = 1
            Y[data_idx] = 0
    

    offset = len(safe_data_addr)
    for data_idx, addr in enumerate(unsafe_data_addr):
        npz_file = np.load(file_address_prefix + addr)
        positions_and_orientations = npz_file['positions_and_orientations']
        for idx, p_o in enumerate(positions_and_orientations[:2000]):
            X[data_idx+offset, idx, 0:7] = p_o
            X[data_idx+offset, idx, 7] = 1
            
            if data_idx >= 1:
                Y[data_idx+offset] = 0
            else:
                Y[data_idx+offset] = 1
    
    X_train, Y_train = X[:4], Y[:4]
    X_val, Y_val = X[4:], Y[4:]
    max_traj_steps = 2000
    return X_train, Y_train, X_val, Y_val, max_traj_steps, X, Y

# read_panda_data()