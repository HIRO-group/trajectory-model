import os
import csv
import numpy as np
from trajectory_model.spill_free.constants import MAX_TRAJ_STEPS, EMBED_DIM, DT

DIR_PREFIX = '/home/ava/projects/trajectory-model/data/mocap_new/'

# nospill, spill, radius, height, fill_level
FILE_NAMES_NOSPILL_SPILL = \
    [("big/full/spill-free/", "big/full/spilled/", 3, 4, 0.8),
        ("big/half-full/spill-free/", "big/half-full/spilled/", 3, 4, 0.3),
        ("small/full/spill-free/", "small/full/spilled/", 1.8, 5, 0.8),
        ("small/half-full/spill-free/", "small/half-full/spilled/", 1.8, 5, 0.5)]


def read_a_file(file_path, radius, height, fill_level):
    X = np.zeros((1, 10000, EMBED_DIM), dtype=np.float64)
    trajectory_index = 0
    with open(file_path, mode ='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            keys = list(row.values())
            x, y, z = np.float64(keys[1]), np.float64(keys[2]), np.float64(keys[3])
            a, b, c, d = np.float64(keys[4]), np.float64(keys[5]), np.float64(keys[6]), np.float64(keys[7])
            embedding = np.array([[x, y, z, a, b, c, d, radius, height, fill_level]])
            X[0, trajectory_index, :] = embedding
            trajectory_index += 1

    X = X[0, 0:10000:DT, :]
    X = X[0, 0:MAX_TRAJ_STEPS, :]
    return X


def read_a_directory(directory_path, radius, height, fill_level):
    X = np.zeros((1, MAX_TRAJ_STEPS, EMBED_DIM), dtype=np.float64)
    files = os.listdir(directory_path)
    for file in files:
        file_path = directory_path + file
        X_new = read_a_file(file_path, radius, height, fill_level) # single traj
        X = np.concatenate((X, X_new), axis=0)
    return X


def handle_nospill(nospill_file, radius, height, fill_level):
    file_path_no_spill = DIR_PREFIX + nospill_file
    X = read_a_directory(file_path_no_spill, radius, height, fill_level) # multiple trajs
    Y = np.zeros((X.shape[0], 1))
    return X, Y


def handle_spill(spill_file, radius, height, fill_level):
    file_path_spill = DIR_PREFIX + spill_file
    X = read_a_directory(file_path_spill, radius, height, fill_level) # multiple trajs
    Y = np.ones((X.shape[0], 1))
    return X, Y


def read_from_files():
    X = np.zeros((1, MAX_TRAJ_STEPS, EMBED_DIM), dtype=np.float64)
    Y = np.zeros((1, 1))

    for row in FILE_NAMES_NOSPILL_SPILL:
        nospill_file, spill_file = row[0], row[1]
        radius, height, fill_level = row[2], row[3], row[4]

        X_ns, Y_ns = handle_nospill(nospill_file, radius, height, fill_level)
        X = np.concatenate((X, X_ns), axis=0)
        Y = np.concatenate((Y, Y_ns), axis=0)

        X_s, Y_s = handle_spill(spill_file, radius, height, fill_level)
        X = np.concatenate((X, X_s), axis=0)
        Y = np.concatenate((Y, Y_s), axis=0)    
    return X, Y

def copy_last_non_zero_value(X):
    for e_id in range(X.shape[0]):
        embedding = X[e_id, :, 0:7] 
        all_zero_indices = np.where(np.all(embedding == 0, axis=1))
        if len(all_zero_indices[0]) == 0:
            continue
        first_zero_index =  all_zero_indices[0][0]
        
        if first_zero_index == 0:
            X[e_id, :, :] = X[e_id-1, :, :]
        else:
            last_non_zero_index = first_zero_index - 1
            X[e_id, first_zero_index:, :] = X[e_id, last_non_zero_index, :]
    return X


def transform_trajectory(X):
    for e_id in range(X.shape[0]):
        xyz = X[e_id, :, 0:3]
        xyz = xyz - X[e_id, 0, 0:3]
        X[e_id, :, 0:3] = xyz
    return X


def add_equivalent_quaternions(X, Y):
    X_new = np.zeros((2 * X.shape[0], MAX_TRAJ_STEPS, EMBED_DIM), dtype=np.float64)
    Y_new = np.zeros((2 * X.shape[0], 1), dtype=np.float64)

    for e_id in range(0, X.shape[0]):
        X_new[2 * e_id] = X[e_id]
        X_new[2 * e_id + 1] = X[e_id]
        X_new[2 * e_id + 1, :, 3:7] = -X[e_id, :, 3:7]
        Y_new[2 * e_id] = Y[e_id]
        Y_new[2 * e_id + 1] = Y[e_id]
    return X_new, Y_new



def add_partial_trajectory(X, Y):
    num_experiments = X.shape[0]
    data_per_experiment = 4
    new_X_num_data = data_per_experiment * X.shape[0]
    X_new = np.zeros((new_X_num_data, X.shape[1], EMBED_DIM), dtype=np.float64)
    Y_new = np.zeros((new_X_num_data, 1), dtype=np.float64)

    for e_id in range(X.shape[0]):
        for i in range(data_per_experiment):
            Y_new[e_id * data_per_experiment + i, :] = Y[e_id]
            X_new[e_id * data_per_experiment + i, 0: int((i+1) * X.shape[1]/data_per_experiment), :] = X[e_id, 0: int((i+1) * X.shape[1]/data_per_experiment), :]
            # fill the rest with the last value
            X_new[e_id * data_per_experiment + i, int((i+1) * X.shape[1]/data_per_experiment):, :] = X[e_id, int((i+1) * X.shape[1]/data_per_experiment)-1, :]
    return X_new, Y_new


def  add_delta_X(X):
    X_new = np.zeros((X.shape[0], MAX_TRAJ_STEPS, EMBED_DIM), dtype=np.float64)
    for e_id in range(X.shape[0]):
        for i in range(1, X.shape[1]):
            X_new[e_id, i, 0:3] = X[e_id, i, 0:3] - X[e_id, i-1, 0:3]
    return X_new


def add_reverse_X(X, Y):
    X_new = np.zeros((2 * X.shape[0], MAX_TRAJ_STEPS, EMBED_DIM), dtype=np.float64)
    Y_new = np.zeros((2 * X.shape[0], 1), dtype=np.float64)
    for e_id in range(X.shape[0]):
        for i in range(1, X.shape[1]):
            X_new[2 * e_id, i, 0:3] = X[e_id, i, 0:3]
            X_new[2 * e_id + 1, i, 0:3] = - X[e_id, i, 0:3]
            
            Y_new[2 * e_id] = Y[e_id]
            Y_new[2 * e_id + 1] = Y[e_id]


def process_data():
    X, Y = read_from_files()
    X = copy_last_non_zero_value(X)
    X = transform_trajectory(X)
    X, Y = add_equivalent_quaternions(X, Y)
    X, Y = add_partial_trajectory(X, Y)
    X = add_delta_X(X)
    X, Y = add_reverse_X(X, Y)


if __name__ == "__main__":
    X, Y = process_data()
