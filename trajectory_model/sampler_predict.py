import numpy as np
from trajectory_model.process_data import read_from_files, copy_last_non_zero_value, \
    transform_trajectory, add_equivalent_quaternions, round_down_orientation_and_pos
from trajectory_model.spill_free.constants import EMBED_DIM
from trajectory_model.helper import plot_X, plot_multiple_e_ids, plot_multiple_X

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
    

def transform_for_IK(X):
    pass


X, Y = read_from_files()
X, Y = keep_spill_free(X, Y)
X = copy_last_non_zero_value(X)
X = transform_trajectory(X)
X = transform_for_IK(X)
X, Y = add_equivalent_quaternions(X, Y)
X = round_down_orientation_and_pos(X)
X = select_waypoints(X)


def sample_state(trajectory):
    # Should sample from the specific cup properties
    random_e_id = np.random.randint(0, X.shape[0])
    random_traj_step = np.random.randint(0, X.shape[1])
    return X[random_e_id, random_traj_step, 0:7]


xyzabcd = sample_state(0)
print("xyzabcd: ", xyzabcd)