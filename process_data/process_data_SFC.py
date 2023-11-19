import os
import csv
import numpy as np
from trajectory_model.spill_free.constants import \
    MAX_TRAJ_STEPS, EMBED_DIM, EMBED_LOC, EMBED_PROP, MOCAP_DT, BLANK_VAL, \
    BIG_RADIUS_B, BIG_HEIGHT, BIG_RADIUS_U, BIG_FILL_80, BIG_FILL_30, \
    SMALL_RADIUS_B, SMALL_HEIGHT, SMALL_RADIUS_U, SMALL_FILL_80, SMALL_FILL_50, \
    SHORT_TUMBLER_RADIUS_B, SHORT_TUMBLER_HEIGHT, SHORT_TUMBLER_RADIUS_U, SHORT_TUMBLER_FILL_30, SHORT_TUMBLER_FILL_70, \
    TALL_TUMBLER_RADIUS_B, TALL_TUMBLER_HEIGHT, TALL_TUMBLER_RADIUS_U, TALL_TUMBLER_FILL_50, TALL_TUMBLER_FILL_80, \
    TUMBLER_RADIUS_B, TUMBLER_HEIGHT, TUMBLER_RADIUS_U, TUMBLER_FILL_30, TUMBLER_FILL_70, \
    WINE_RADIUS_B, WINE_HEIGHT, WINE_RADIUS_U, WINE_FILL_30, WINE_FILL_70
from trajectory_model.helper.helper import plot_X, plot_multiple_e_ids, plot_multiple_X
from trajectory_model.helper.read import read_panda_vectors
from trajectory_model.classifier_predict_func_api import process_panda_to_model_input

DIR_PREFIX = '/home/ava/projects/trajectory-model/data/'

# nospill, spill, radius_buttom, height, radius_top, fill_level
FILE_NAMES_NOSPILL_SPILL = \
    [("mocap_new/big/80/spill-free/", "mocap_new/big/80/spilled/",
    BIG_RADIUS_B, BIG_HEIGHT, BIG_RADIUS_U, BIG_FILL_80),

     ("mocap_new/big/30/spill-free/", "mocap_new/big/30/spilled/",
    BIG_RADIUS_B, BIG_HEIGHT, BIG_RADIUS_U, BIG_FILL_30),

     ("mocap_new/small/80/spill-free/", "mocap_new/small/80/spilled/", 
    SMALL_RADIUS_B, SMALL_HEIGHT, SMALL_RADIUS_U, SMALL_FILL_80),

     ("mocap_new/small/50/spill-free/", "mocap_new/small/50/spilled/", 
    SMALL_RADIUS_B, SMALL_HEIGHT, SMALL_RADIUS_U, SMALL_FILL_50),

     ("mocap_new_cups/shorttumbler/30/spill-free/", "mocap_new_cups/shorttumbler/30/spilled/",
    SHORT_TUMBLER_RADIUS_B, SHORT_TUMBLER_HEIGHT, SHORT_TUMBLER_RADIUS_U, SHORT_TUMBLER_FILL_30),
       
     ("mocap_new_cups/shorttumbler/70/spill-free/", "mocap_new_cups/shorttumbler/70/spilled/",
    SHORT_TUMBLER_RADIUS_B, SHORT_TUMBLER_HEIGHT, SHORT_TUMBLER_RADIUS_U, SHORT_TUMBLER_FILL_70),
     
     ("mocap_new_cups/talltumbler/50/spill-free/", "mocap_new_cups/talltumbler/50/spilled/",
    TALL_TUMBLER_RADIUS_B, TALL_TUMBLER_HEIGHT, TALL_TUMBLER_RADIUS_U, TALL_TUMBLER_FILL_50),
    
     ("mocap_new_cups/talltumbler/80/spill-free/", "mocap_new_cups/talltumbler/80/spilled/",
    TALL_TUMBLER_RADIUS_B, TALL_TUMBLER_HEIGHT, TALL_TUMBLER_RADIUS_U, TALL_TUMBLER_FILL_80),
     
     ("mocap_new_cups/tumbler/30/spill-free/", "mocap_new_cups/tumbler/30/spilled/",
    TUMBLER_RADIUS_B, TUMBLER_HEIGHT, TUMBLER_RADIUS_U, TUMBLER_FILL_30),

     ("mocap_new_cups/tumbler/70/spill-free/", "mocap_new_cups/tumbler/70/spilled/",
    TUMBLER_RADIUS_B, TUMBLER_HEIGHT, TUMBLER_RADIUS_U, TUMBLER_FILL_70),
     
     ("mocap_new_cups/wineglass/30/spill-free/", "mocap_new_cups/wineglass/30/spilled/",
    WINE_RADIUS_B, WINE_HEIGHT, WINE_RADIUS_U, WINE_FILL_30),
     
    ("mocap_new_cups/wineglass/70/spill-free/", "mocap_new_cups/wineglass/70/spilled/",
    WINE_RADIUS_B, WINE_HEIGHT, WINE_RADIUS_U, WINE_FILL_70),]


FILE_NAMES_AUGMENT_SPILL = [
    ("mocap_new/big/30/spilled/", [(BIG_RADIUS_B, BIG_HEIGHT, BIG_RADIUS_U, BIG_FILL_80)]),
    ("mocap_new/small/50/spilled/", [(SMALL_RADIUS_B, SMALL_HEIGHT, SMALL_RADIUS_U, SMALL_FILL_80)]),
    ("mocap_new_cups/shorttumbler/30/spilled/", [(SHORT_TUMBLER_RADIUS_B, SHORT_TUMBLER_HEIGHT, SHORT_TUMBLER_RADIUS_U, SHORT_TUMBLER_FILL_70)]),
    ("mocap_new_cups/talltumbler/50/spilled/", [(TALL_TUMBLER_RADIUS_B, TALL_TUMBLER_HEIGHT, TALL_TUMBLER_RADIUS_U, TALL_TUMBLER_FILL_80)]),
    ("mocap_new_cups/tumbler/30/spilled/", [(TUMBLER_RADIUS_B, TUMBLER_HEIGHT, TUMBLER_RADIUS_U, TUMBLER_FILL_70)]),
    ("mocap_new_cups/wineglass/30/spilled/", [(WINE_RADIUS_B, WINE_HEIGHT, WINE_RADIUS_U, WINE_FILL_70)]),
]

FILE_NAMES_AUGMENT_SPILLFREE = [
    ("mocap_new/big/80/spill-free/", [(BIG_RADIUS_B, BIG_HEIGHT, BIG_RADIUS_U, BIG_FILL_30)]),
    ("mocap_new/small/80/spill-free/", [(SMALL_RADIUS_B, SMALL_HEIGHT, SMALL_RADIUS_U, SMALL_FILL_50)]),
    ("mocap_new_cups/shorttumbler/70/spill-free/", [(SHORT_TUMBLER_RADIUS_B, SHORT_TUMBLER_HEIGHT, SHORT_TUMBLER_RADIUS_U, SHORT_TUMBLER_FILL_30)]),
    ("mocap_new_cups/talltumbler/80/spill-free/", [(TALL_TUMBLER_RADIUS_B, TALL_TUMBLER_HEIGHT, TALL_TUMBLER_RADIUS_U, TALL_TUMBLER_FILL_50)]),
    ("mocap_new_cups/tumbler/70/spill-free/", [(TUMBLER_RADIUS_B, TUMBLER_HEIGHT, TUMBLER_RADIUS_U, TUMBLER_FILL_30)]),
    ("mocap_new_cups/wineglass/70/spill-free/", [(WINE_RADIUS_B, WINE_HEIGHT, WINE_RADIUS_U, WINE_FILL_30)]),
]

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


def read_a_file(file_path, radius_b, height, radius_u, fill_level):
    X = np.zeros((1, 2000, EMBED_DIM), dtype=np.float64)
    trajectory_index = 0

    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            keys = list(row.values())
            y, x, z = np.float64(keys[1]), np.float64(keys[2]), np.float64(keys[3])
            a, b, c, d = np.float64(keys[4]), np.float64(keys[5]), np.float64(keys[6]), np.float64(keys[7])
            embedding = np.array([[x, y, z, a, b, c, d, radius_b, height, radius_u, fill_level]])
            X[0, trajectory_index, :] = embedding
            trajectory_index += 1
    
    X = trim_noise(X)

    X = X[:, 0:2000:MOCAP_DT, :]
    X = X[:, 0:MAX_TRAJ_STEPS, :]

    if X.shape[1] < MAX_TRAJ_STEPS:
        blank_vals = BLANK_VAL * np.ones((1, MAX_TRAJ_STEPS - X.shape[1], EMBED_DIM), dtype=np.float64) 
        X = np.concatenate((X, blank_vals), axis=1)

    return X


def read_a_directory(directory_path, radius_b, height, radius_u, fill_level):
    X = np.zeros((0, MAX_TRAJ_STEPS, EMBED_DIM), dtype=np.float64)
    files = os.listdir(directory_path)
    for file in files:
        file_path = directory_path + file
        X_new = read_a_file(file_path, radius_b, height, radius_u, fill_level)  # single traj
        X = np.concatenate((X, X_new), axis=0)
    return X


def handle_nospill(nospill_file, radius_b, height, radius_u, fill_level):
    file_path_no_spill = DIR_PREFIX + nospill_file
    X = read_a_directory(file_path_no_spill, radius_b, height, radius_u, fill_level)  # multiple trajs
    Y = np.zeros((X.shape[0], 1))
    return X, Y


def handle_spill(spill_file, radius_b, height, radius_u, fill_level):
    file_path_spill = DIR_PREFIX + spill_file
    X = read_a_directory(file_path_spill, radius_b, height, radius_u, fill_level)  # multiple trajs
    Y = np.ones((X.shape[0], 1))
    return X, Y


def read_from_files(file_list=FILE_NAMES_NOSPILL_SPILL):
    X = np.zeros((0, MAX_TRAJ_STEPS, EMBED_DIM), dtype=np.float64)
    Y = np.zeros((0, 1))

    for row in file_list:
        nospill_file, spill_file = row[0], row[1]
        radius_b, height, radius_u, fill_level = row[2], row[3], row[4], row[5]

        X_ns, Y_ns = handle_nospill(nospill_file, radius_b, height, radius_u, fill_level)
        X = np.concatenate((X, X_ns), axis=0)
        Y = np.concatenate((Y, Y_ns), axis=0)

        X_s, Y_s = handle_spill(spill_file, radius_b, height, radius_u, fill_level)
        X = np.concatenate((X, X_s), axis=0)
        Y = np.concatenate((Y, Y_s), axis=0)
    return X, Y


def augment_data(X, Y):
    for row in FILE_NAMES_AUGMENT_SPILL:
        spill_file = row[0]
        for prop in row[1]:
            radius_b, height, radius_u, fill_level = prop[0], prop[1], prop[2], prop[3]
            X_s, Y_s = handle_spill(spill_file, radius_b, height, radius_u, fill_level)
            X = np.concatenate((X, X_s), axis=0)
            Y = np.concatenate((Y, Y_s), axis=0)

    for row in FILE_NAMES_AUGMENT_SPILLFREE:
        spillfree_file = row[0]
        for prop in row[1]:
            radius_b, height, radius_u, fill_level = prop[0], prop[1], prop[2], prop[3]
            X_s, Y_s = handle_nospill(spillfree_file, radius_b, height, radius_u, fill_level)
            X = np.concatenate((X, X_s), axis=0)
            Y = np.concatenate((Y, Y_s), axis=0)
    
    return X, Y


def fill_with_blanks(X):
    for e_id in range(X.shape[0]):
        embedd_loc = X[e_id, :, 0:EMBED_LOC]
        all_zero_indices = np.where(np.all(embedd_loc == 0, axis=1))
        if len(all_zero_indices[0]) == 0:
            continue
        first_zero_index = all_zero_indices[0][0]

        if first_zero_index == 0:
            X[e_id, :, :] = X[e_id-1, :, :]
        else:
            X[e_id, first_zero_index:, :] = [BLANK_VAL for _ in range(EMBED_DIM)]
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
    delta_X = np.copy(X)
    for e_id in range(X.shape[0]):
        for i in range(1, X.shape[1]):
            delta_X[e_id, i, 0:3] = X[e_id, i, 0:3] - X[e_id, i-1, 0:3]
    return delta_X

def process_panda_file(X, Y, filenames, spill_free):
    for fps in filenames:
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
            if spill_free:
                Y = np.concatenate((Y, np.zeros((1, 1))), axis=0)
            else:
                Y = np.concatenate((Y, np.ones((1, 1))), axis=0)
    return X, Y


def add_panda_trajectories(X, Y):
    filenames_props_NOSPILL = [
        (['01-09-2023 13-42-14', '01-09-2023 13-58-43', '01-09-2023 14-09-56'], 
        np.array([[BIG_RADIUS_B, BIG_HEIGHT, BIG_RADIUS_U, BIG_FILL_30]])),
        
        (['10-09-2023 10-03-18', '10-09-2023 10-06-37', '10-09-2023 13-10-26',
        '10-09-2023 13-14-07'],
        np.array([[SMALL_RADIUS_B, SMALL_HEIGHT, SMALL_RADIUS_U, SMALL_FILL_80]])),
        
        (['10-09-2023 13-30-09', '10-09-2023 13-32-29', '10-09-2023 13-39-37'], 
        np.array([[SMALL_RADIUS_B, SMALL_HEIGHT, SMALL_RADIUS_U, SMALL_FILL_50]]))
    ]

    file_names_props_SPILL = [
        (['13-11-2023 13-17-10', '13-11-2023 13-19-03', '13-11-2023 13-22-02',
          '13-11-2023 13-23-06', '13-11-2023 13-25-15', '13-11-2023 13-26-22',
          '13-11-2023 15-52-06', '13-11-2023 16-02-34', '13-11-2023 16-33-28',
          '13-11-2023 16-41-08', '13-11-2023 16-45-30', '13-11-2023 16-46-41',
          '13-11-2023 16-47-52', '13-11-2023 16-49-54', '13-11-2023 17-00-38',
          '13-11-2023 17-01-48', '13-11-2023 17-03-14', '13-11-2023 17-06-53',
          '13-11-2023 17-08-32', '13-11-2023 17-09-56', '13-11-2023 17-11-11',
          '13-11-2023 17-20-14', '13-11-2023 17-23-14', '13-11-2023 17-27-18',
          '13-11-2023 17-30-59', '13-11-2023 17-32-32', '13-11-2023 17-33-44'], 
          np.array([[BIG_RADIUS_B, BIG_HEIGHT, BIG_RADIUS_U, BIG_FILL_30]]))
    ]

    X, Y = process_panda_file(X, Y, filenames_props_NOSPILL, spill_free=True)
    X, Y = process_panda_file(X, Y, file_names_props_SPILL, spill_free=False)

    return X, Y


def process_data_SFC():
    X, Y = read_from_files()
    X, Y = augment_data(X, Y)
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
    plot_multiple_e_ids(X, [0], 0.01)
