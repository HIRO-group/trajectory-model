import numpy as np
from trajectory_model.process_data import read_from_files, copy_last_non_zero_value, \
    transform_trajectory, add_equivalent_quaternions, round_down_orientation_and_pos, \
    add_reverse_X
from trajectory_model.spill_free.constants import EMBED_DIM, BIG_RADIUS, BIG_HEIGHT, SMALL_RADIUS, \
    SMALL_HEIGHT, BIG_FILL_FULL, BIG_FILL_HALF, SMALL_FILL_FULL, SMALL_FILL_HALF
from trajectory_model.helper import plot_X, plot_multiple_e_ids, plot_multiple_X
from trajectory_model.rotate_quaternion import quaternion_to_angle_axis, rotate_quaternion


MAX_NUM_WAYPOINTS = 10

def keep_spill_free(X, Y):
    X_new, Y_new = [], []
    for e_id in range(X.shape[0]):
        if Y[e_id] == 0:
            X_new.append(X[e_id])
            Y_new.append(Y[e_id])
    X_new = np.array(X_new)
    Y_new = np.array(Y_new)
    return X_new, Y_new

def select_waypoints(X):
    X_new = np.zeros((X.shape[0], MAX_NUM_WAYPOINTS, EMBED_DIM), dtype=np.float64)
    for e_id in range(X.shape[0]):
        embedding = X[e_id, :, 0:7]
        all_zero_indices = np.where(np.all(embedding == 0, axis=1))

        if  len(all_zero_indices[0]) != 0  and all_zero_indices[0][0] == 0:
            X_new[e_id] = X_new[e_id - 1]
            continue

        if len(all_zero_indices[0]) == 0:
            max_index = X.shape[1] - 1
        else:
            max_index = all_zero_indices[0][0] - 1 # The last non zero index

        step_size = int(max_index / MAX_NUM_WAYPOINTS)
        remainder = max_index % MAX_NUM_WAYPOINTS
        X_new[e_id] = np.array([np.array(X[e_id, i, :]) for i in range(0, max_index - remainder, step_size)])

        all_zero_indices = np.where(np.all(X_new[e_id, :, 3:7] == 0, axis=1))
        if len(all_zero_indices[0]) == 0:
            continue
        else:
            last_non_zero_index = all_zero_indices[0][0] - 1
            X_new[e_id, last_non_zero_index + 1:, :] = X_new[e_id, last_non_zero_index, :]
    return X_new
    
def translate_for_IK(X):
    for e_id in range(X.shape[0]):
        min_x = np.min(X[e_id, :, 0])
        min_y = np.min(X[e_id, :, 1])
        min_z = np.min(X[e_id, :, 2])

        X[e_id, :, 0] = X[e_id, :, 0] - min_x + 0.2
        X[e_id, :, 1] = X[e_id, :, 1] - min_y - 0.6
        X[e_id, :, 2] = X[e_id, :, 2] - min_z + 0.4
    return X


def rotate_for_IK(X):
    q_start = (-0.0045020487159490585, -0.001336313085630536, -0.21967099606990814, -0.9755628108978271)
    q = (0.7258497516707821, -0.6872946206328441, -0.027717186881415688, -0.6877996017242218)
    angle, axis = quaternion_to_angle_axis(q, q_start)

    for e_id in range(X.shape[0]):
        a, b, c, d = X[e_id, :, 3], X[e_id, :, 4], X[e_id, :, 5], X[e_id, :, 6]
        x_rot, y_rot, z_rot, w_rot = rotate_quaternion((a, b, c, d), angle, axis)
        X[e_id, :, 3], X[e_id, :, 4], X[e_id, :, 5], X[e_id, :, 6] = x_rot, y_rot, z_rot, w_rot
    return X


def transform_for_IK(X):
    X = translate_for_IK(X)
    X = rotate_for_IK(X)
    return X

# FILE_NAMES_NOSPILL_SPILL = \
#     [("big/full/spill-free/", "big/full/spilled/", BIG_RADIUS, BIG_HEIGHT, BIG_FILL_FULL),
#         ("big/half-full/spill-free/", "big/half-full/spilled/", BIG_RADIUS, BIG_HEIGHT, BIG_FILL_HALF),
#         ("small/full/spill-free/", "small/full/spilled/", SMALL_RADIUS, SMALL_HEIGHT, SMALL_FILL_FULL),
#         ("small/half-full/spill-free/", "small/half-full/spilled/", SMALL_RADIUS, SMALL_HEIGHT, SMALL_FILL_HALF)]

file_list = [("big/full/spill-free/", "big/full/spilled/", BIG_RADIUS, BIG_HEIGHT, BIG_FILL_HALF)]

X, Y = read_from_files(file_list)
X, Y = keep_spill_free(X, Y)
X = copy_last_non_zero_value(X)
X = transform_trajectory(X)
X = add_reverse_X(X)
X = transform_for_IK(X)
X, Y = add_equivalent_quaternions(X, Y)
X = round_down_orientation_and_pos(X)
X = select_waypoints(X)


def sample_state(trajectory):
    random_e_id = np.random.randint(0, X.shape[0])
    random_traj_step = np.random.randint(0, X.shape[1])
    return X[random_e_id, random_traj_step, 0:7]

# plot_X(X, 0, 0.1)