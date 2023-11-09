import os
import csv
import numpy as np
from trajectory_model.spill_free.constants import MAX_TRAJ_STEPS, EMBED_DIM, MOCAP_DT, \
    BIG_RADIUS, BIG_HEIGHT, SMALL_RADIUS, SMALL_HEIGHT, \
    BIG_FILL_FULL, BIG_FILL_HALF, SMALL_FILL_FULL, SMALL_FILL_HALF, BLANK_VAL
from trajectory_model.helper.helper import plot_X, plot_multiple_e_ids, plot_multiple_X
from trajectory_model.helper.read import read_panda_vectors
from trajectory_model.classifier_predict_func_api import process_panda_to_model_input

DIR_PREFIX = '/home/ava/projects/trajectory-model/data/mocap_new/'


# nospill, spill, radius, height, fill_level
FILE_NAMES_NOSPILL_SPILL = \
    [("big/full/spill-free/", "big/full/spilled/", BIG_RADIUS, BIG_HEIGHT, BIG_FILL_FULL),
        ("big/half-full/spill-free/", "big/half-full/spilled/", BIG_RADIUS, BIG_HEIGHT, BIG_FILL_HALF),
        ("small/full/spill-free/", "small/full/spilled/", SMALL_RADIUS, SMALL_HEIGHT, SMALL_FILL_FULL),
        ("small/half-full/spill-free/", "small/half-full/spilled/", SMALL_RADIUS, SMALL_HEIGHT, SMALL_FILL_HALF)]


def trim_noise(X):
    diffs_pos = []
    diffs_ori = []

    for step in range(1, X.shape[1]):
        prev_pos = X[0, step-1, 0:3]
        curr_pos = X[0, step, 0:3]

        prev_ori = X[0, step-1, 3:7]
        curr_ori = X[0, step, 3:7]
        
        diff_pos = np.linalg.norm(curr_pos - prev_pos)
        diff_ori = np.linalg.norm(curr_ori - prev_ori)

        diffs_ori.append(diff_ori)
        diffs_pos.append(diff_pos)

    # find the first non_zero diff
    first_non_zero_index = 0
    for i in range(len(diffs_pos)):
        if diffs_pos[i] > 0:
            first_non_zero_index = i
            break
    for i in range(len(diffs_ori)):
        if diffs_ori[i] > 0:
            first_non_zero_index = min(first_non_zero_index, i)
            break

    # find the last non_zero diff
    last_non_zero_index = 0
    for i in range(len(diffs_pos)-1, -1, -1):
        if diffs_pos[i] > 0:
            last_non_zero_index = i
            break
    for i in range(len(diffs_ori)-1, -1, -1):
        if diffs_ori[i] > 0:
            last_non_zero_index = max(last_non_zero_index, i)
            break

    X = X[:, first_non_zero_index:last_non_zero_index, :]
    return X


def read_a_file(file_path, radius, height, fill_level):
    X = np.zeros((1, 2000, EMBED_DIM), dtype=np.float64)
    trajectory_index = 0

    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            keys = list(row.values())
            y, x, z = np.float64(keys[1]), np.float64(keys[2]), np.float64(keys[3])
            a, b, c, d = np.float64(keys[4]), np.float64(keys[5]), np.float64(keys[6]), np.float64(keys[7])
            embedding = np.array([[x, y, z, a, b, c, d, radius, height, fill_level]])
            X[0, trajectory_index, :] = embedding
            trajectory_index += 1
    
    X = trim_noise(X)

    X = X[:, 0:2000:MOCAP_DT, :]
    X = X[:, 0:MAX_TRAJ_STEPS, :]

    if X.shape[1] < MAX_TRAJ_STEPS:
        blank_vals = BLANK_VAL * np.ones((1, MAX_TRAJ_STEPS - X.shape[1], EMBED_DIM), dtype=np.float64) 
        X = np.concatenate((X, blank_vals), axis=1)

    return X


def read_a_directory(directory_path, radius, height, fill_level):
    X = np.zeros((0, MAX_TRAJ_STEPS, EMBED_DIM), dtype=np.float64)
    files = os.listdir(directory_path)
    for file in files:
        file_path = directory_path + file
        X_new = read_a_file(file_path, radius, height, fill_level)  # single traj
        X = np.concatenate((X, X_new), axis=0)
    return X


def handle_nospill(nospill_file, radius, height, fill_level):
    file_path_no_spill = DIR_PREFIX + nospill_file
    X = read_a_directory(file_path_no_spill, radius, height, fill_level)  # multiple trajs
    Y = np.zeros((X.shape[0], 1))
    return X, Y


def handle_spill(spill_file, radius, height, fill_level):
    file_path_spill = DIR_PREFIX + spill_file
    X = read_a_directory(file_path_spill, radius, height, fill_level)  # multiple trajs
    Y = np.ones((X.shape[0], 1))
    return X, Y


def read_from_files(file_list = FILE_NAMES_NOSPILL_SPILL):
    X = np.zeros((0, MAX_TRAJ_STEPS, EMBED_DIM), dtype=np.float64)
    Y = np.zeros((0, 1))

    for row in file_list:
        nospill_file, spill_file = row[0], row[1]
        radius, height, fill_level = row[2], row[3], row[4]

        X_ns, Y_ns = handle_nospill(nospill_file, radius, height, fill_level)
        X = np.concatenate((X, X_ns), axis=0)
        Y = np.concatenate((Y, Y_ns), axis=0)

        X_s, Y_s = handle_spill(spill_file, radius, height, fill_level)
        X = np.concatenate((X, X_s), axis=0)
        Y = np.concatenate((Y, Y_s), axis=0)

    return X, Y


def fill_with_blanks(X):
    # print(X.shape)
    for e_id in range(X.shape[0]):
        embedding = X[e_id, :, 0:7]
        all_zero_indices = np.where(np.all(embedding == 0, axis=1))
        if len(all_zero_indices[0]) == 0:
            continue
        first_zero_index = all_zero_indices[0][0]

        if first_zero_index == 0:
            X[e_id, :, :] = X[e_id-1, :, :]
        else:
            # last_non_zero_index = first_zero_index - 1
            X[e_id, first_zero_index:, :] = [BLANK_VAL, BLANK_VAL, BLANK_VAL,
                                              BLANK_VAL, BLANK_VAL, BLANK_VAL, 
                                              BLANK_VAL, BLANK_VAL, BLANK_VAL, BLANK_VAL]
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
    data_per_experiment = 4
    new_X_num_data = data_per_experiment * X.shape[0]
    X_new = np.zeros((new_X_num_data, X.shape[1], EMBED_DIM), dtype=np.float64)
    Y_new = np.zeros((new_X_num_data, 1), dtype=np.float64)

    for e_id in range(X.shape[0]):
        for i in range(data_per_experiment):
            Y_new[e_id * data_per_experiment + i, :] = Y[e_id]
            X_new[e_id * data_per_experiment + i, 0: int(
                (i+1) * X.shape[1]/data_per_experiment), :] = X[e_id, 0: int((i+1) * X.shape[1]/data_per_experiment), :]
            
            # fill the rest with the last value
            X_new[e_id * data_per_experiment + i, int((i+1) * X.shape[1]/data_per_experiment):, :] = X[e_id, int(
                (i+1) * X.shape[1]/data_per_experiment)-1, :]
    return X_new, Y_new


def add_delta_X(X):
    X_new = np.zeros((X.shape[0], MAX_TRAJ_STEPS, EMBED_DIM), dtype=np.float64)
    for e_id in range(X.shape[0]):
        for i in range(1, X.shape[1]):
            X_new[e_id, i, 0:3] = X[e_id, i, 0:3] - X[e_id, i-1, 0:3]
    return X_new


def reverse_y_axis(X):
    X[:, :, 1] = -X[:, :, 1]
    return X


def round_down_orientation_and_pos(X):
    for e_id in range(X.shape[0]):
        for i in range(X.shape[1]):
            X[e_id, i, 0:3] = np.round(X[e_id, i, 0:3], 2)
            X[e_id, i, 3:7] = np.round(X[e_id, i, 3:7], 2)
    return X


def keep_spill_free(X, Y):
    X_new, Y_new = [], []
    for e_id in range(X.shape[0]):
        if Y[e_id] == 0:
            X_new.append(X[e_id])
            Y_new.append(Y[e_id])
    X_new = np.array(X_new)
    Y_new = np.array(Y_new)
    return X_new, Y_new


def compute_delta_X(X):
    # compute delta x
    # delta_X = np.zeros((X.shape[0], MAX_TRAJ_STEPS, X.shape[2]), dtype=np.float64)
    delta_X = np.copy(X)
    for e_id in range(X.shape[0]):
        for i in range(1, X.shape[1]):
            delta_X[e_id, i, 0:3] = X[e_id, i, 0:3] - X[e_id, i-1, 0:3]
            # delta_X[e_id, i, 3:]  = X[e_id, i, 3:]
    return delta_X

def add_panda_trajectories(X, Y):
    filenames_props_no_spill = [
        (['01-09-2023 13-42-14', '01-09-2023 13-58-43', '01-09-2023 14-09-56'], 
                            np.array([[BIG_RADIUS, BIG_HEIGHT, BIG_FILL_HALF]])),
        (['10-09-2023 10-03-18', '10-09-2023 10-06-37', '10-09-2023 13-10-26', '10-09-2023 13-14-07'],
                            np.array([[SMALL_RADIUS, SMALL_HEIGHT, SMALL_FILL_FULL]])),
        (['10-09-2023 13-30-09', '10-09-2023 13-32-29', '10-09-2023 13-39-37'], 
                            np.array([[SMALL_RADIUS, SMALL_HEIGHT, SMALL_FILL_HALF]]))
    ]
    for fps in filenames_props_no_spill:
        filenames = fps[0]
        properties = fps[1]
        properties = np.repeat(properties, X.shape[1], axis=0)
        for fn in filenames:
            panda_file_path =  '/home/ava/projects/assets/cartesian/'+fn+'/cartesian_positions.bin'
            vectors = read_panda_vectors(panda_file_path)

            panda_traj = process_panda_to_model_input(vectors)
            panda_traj = np.concatenate((panda_traj, properties), axis=1)
            panda_traj = panda_traj[np.newaxis, :, :]
            
            X = np.concatenate((X, panda_traj), axis=0)
            Y = np.concatenate((Y, np.zeros((1, 1))), axis=0)

    return X, Y


def process_data_SFC():
    X, Y = read_from_files()
    X = fill_with_blanks(X)
    X = transform_trajectory(X)
    X, Y = add_equivalent_quaternions(X, Y)
    X = round_down_orientation_and_pos(X)
    # X, Y = add_partial_trajectory(X, Y)
    X = reverse_y_axis(X)
    X = compute_delta_X(X)
    X, Y = add_panda_trajectories(X, Y)
    return X, Y


if __name__ == "__main__":
    X, Y = process_data_SFC()
    plot_multiple_e_ids(X, [0, 10, 20, 30, 40], 0.05)
