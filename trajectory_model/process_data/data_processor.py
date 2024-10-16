import os
import csv
import numpy as np
from trajectory_model.SFC.constants import MAX_TRAJ_STEPS, EMBED_DIM, EMBED_LOC, MOCAP_DT, BLANK_VAL
from trajectory_model.process_data.containers import WineGlass, FluteGlass, BasicGlass, RibbedCup, TallCup, CurvyWineGlass
from trajectory_model.process_data.panda_helper import process_panda_to_model_input


DIR_PREFIX = 'data/'

FILE_NAMES_NOSPILL_SPILL = \
    [("mocap/wine_glass/80/spill-free/", "mocap/wine_glass/80/spilled/", WineGlass(WineGlass.high_fill)),
     ("mocap/wine_glass/30/spill-free/", "mocap/wine_glass/30/spilled/", WineGlass(WineGlass.low_fill)),

     ("mocap/flute_glass/80/spill-free/", "mocap/flute_glass/80/spilled/", FluteGlass(FluteGlass.high_fill)),
     ("mocap/flute_glass/50/spill-free/", "mocap/flute_glass/50/spilled/", FluteGlass(FluteGlass.low_fill)),

     ("mocap/ribbed_cup/30/spill-free/", "mocap/ribbed_cup/30/spilled/", RibbedCup(RibbedCup.low_fill)),
     ("mocap/ribbed_cup/70/spill-free/", "mocap/ribbed_cup/70/spilled/", RibbedCup(RibbedCup.high_fill)),
     
     ("mocap/tall_cup/50/spill-free/", "mocap/tall_cup/50/spilled/", TallCup(TallCup.low_fill)),
     ("mocap/tall_cup/80/spill-free/", "mocap/tall_cup/80/spilled/", TallCup(TallCup.high_fill)),
     
     ("mocap/basic_glass/30/spill-free/", "mocap/basic_glass/30/spilled/", BasicGlass(BasicGlass.low_fill)),
     ("mocap/basic_glass/70/spill-free/", "mocap/basic_glass/70/spilled/", BasicGlass(BasicGlass.high_fill)),
     
     ("mocap/curvy_wine_glass/30/spill-free/", "mocap/curvy_wine_glass/30/spilled/", CurvyWineGlass(CurvyWineGlass.low_fill)),
    ("mocap/curvy_wine_glass/70/spill-free/", "mocap/curvy_wine_glass/70/spilled/", CurvyWineGlass(CurvyWineGlass.high_fill))]


FILE_NAMES_AUGMENT_SPILL = [
    ("mocap/wine_glass/30/spilled/", [WineGlass(WineGlass.high_fill)]),
    ("mocap/flute_glass/50/spilled/", [FluteGlass(FluteGlass.high_fill)]),
    ("mocap/ribbed_cup/30/spilled/", [RibbedCup(RibbedCup.high_fill)]),
    ("mocap/tall_cup/50/spilled/", [TallCup(TallCup.high_fill)]),
    ("mocap/basic_glass/30/spilled/", [BasicGlass(BasicGlass.high_fill)]),
    ("mocap/curvy_wine_glass/30/spilled/", [CurvyWineGlass(CurvyWineGlass.high_fill)]),
]

FILE_NAMES_AUGMENT_SPILLFREE = [
    ("mocap/wine_glass/80/spill-free/", [WineGlass(WineGlass.low_fill)]),
    ("mocap/flute_glass/80/spill-free/", [FluteGlass(FluteGlass.low_fill)]),
    ("mocap/ribbed_cup/70/spill-free/", [RibbedCup(RibbedCup.low_fill)]),
    ("mocap/tall_cup/80/spill-free/", [TallCup(TallCup.low_fill)]),
    ("mocap/basic_glass/70/spill-free/", [BasicGlass(BasicGlass.low_fill)]),
    ("mocap/curvy_wine_glass/70/spill-free/", [CurvyWineGlass(CurvyWineGlass.low_fill)]),
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
            a, b, c, w = np.float64(keys[4]), np.float64(keys[5]), np.float64(keys[6]), np.float64(keys[7])
            embedding = np.array([[x, y, z, a, b, c, w, radius_b, height, radius_u, fill_level]])
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
        nospill_file, spill_file, container = row[0], row[1], row[2]
        radius_b, height, radius_u, fill_level = container.diameter_b, container.height, container.diameter_u, container.fill_level

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
        for container in row[1]:
            radius_b, height, radius_u, fill_level = container.diameter_b, container.height, container.diameter_u, container.fill_level
            X_s, Y_s = handle_spill(spill_file, radius_b, height, radius_u, fill_level)
            X = np.concatenate((X, X_s), axis=0)
            Y = np.concatenate((Y, Y_s), axis=0)

    for row in FILE_NAMES_AUGMENT_SPILLFREE:
        spillfree_file = row[0]
        for container in row[1]:
            radius_b, height, radius_u, fill_level = container.diameter_b, container.height, container.diameter_u, container.fill_level
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


def read_panda_trajectory(file_path):
    trajectory = []
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            keys = list(row.values())
            x, y, z = np.float64(keys[0]), np.float64(keys[1]), np.float64(keys[2])
            a, b, c, w = np.float64(keys[3]), np.float64(keys[4]), np.float64(keys[5]), np.float64(keys[6])
            trajectory.append([x, y, z, a, b, c, w])
    return trajectory


def process_panda_file(X, Y, filenames, spill_free):
    for fps in filenames:
        filenames, container = fps[0], fps[1]
        properties = np.array([[container.diameter_b, container.height, container.diameter_u, container.fill_level]])
        properties = np.repeat(properties, X.shape[1], axis=0)

        for fn in filenames:
            panda_file_path =  DIR_PREFIX+'panda/end_effector_space/'+fn+'/cartesian.csv'
            vectors = read_panda_trajectory(panda_file_path)
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
        (['01-09-2023 13-42-14', '01-09-2023 13-58-43', '01-09-2023 14-09-56'], WineGlass(WineGlass.low_fill)),
        (['10-09-2023 10-03-18', '10-09-2023 10-06-37', '10-09-2023 13-10-26',
          '10-09-2023 13-14-07',], FluteGlass(FluteGlass.high_fill)),
        (['10-09-2023 13-30-09', '10-09-2023 13-32-29', '10-09-2023 13-39-37'], FluteGlass(FluteGlass.low_fill)),
        ([], BasicGlass(BasicGlass.high_fill)),
        ([], BasicGlass(BasicGlass.low_fill)),
    ] # 18 

    filenames_props_SPILL = [
        (['13-11-2023 13-17-10', '13-11-2023 13-19-03', '13-11-2023 13-22-02',
          '13-11-2023 13-23-06', '13-11-2023 13-25-15', '13-11-2023 13-26-22',
          '13-11-2023 15-52-06', '13-11-2023 16-02-34', '13-11-2023 16-33-28',
          '13-11-2023 16-41-08', '13-11-2023 16-45-30', '13-11-2023 16-46-41',
          '13-11-2023 16-47-52', '13-11-2023 16-49-54', '13-11-2023 17-00-38',
          '13-11-2023 17-01-48', '13-11-2023 17-03-14', '13-11-2023 17-06-53',
          '13-11-2023 17-08-32', '13-11-2023 17-09-56', '13-11-2023 17-11-11',
          '13-11-2023 17-20-14', '13-11-2023 17-23-14', '13-11-2023 17-27-18',
          '13-11-2023 17-30-59', '13-11-2023 17-32-32', '13-11-2023 17-33-44'], 
          WineGlass(WineGlass.low_fill))
    ] # 27

    # 18 + 27 = 45 panda trajectories

    X, Y = process_panda_file(X, Y, filenames_props_NOSPILL, spill_free=True)
    X, Y = process_panda_file(X, Y, filenames_props_SPILL, spill_free=False)
    return X, Y


def process_data():
    X, Y = read_from_files()
    X, Y = augment_data(X, Y)
    X = fill_with_blanks(X)
    X = transform_trajectory(X)
    X, Y = add_equivalent_quaternions(X, Y)
    X = round_down_orientation_and_pos(X)
    X = reverse_y_axis(X)
    X = compute_delta_X(X)
    X, Y = add_panda_trajectories(X, Y)
    return X, Y


if __name__ == "__main__":
    X, Y = process_data()
