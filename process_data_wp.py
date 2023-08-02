import numpy as np

from trajectory_model.helper import find_significant_curvature_changes, \
    find_significant_orientation_changes, \
    find_significant_position_changes

from trajectory_model.constants import MAX_NUM_WAYPOINTS, MAX_TRAJ_STEPS, EMBED_DIM
from process_data import read_from_mocap_file, transform_trajectory, add_equivalent_quaternions

from trajectory_model.helper import plot_X


# def select_waypoints_hard(X, Y):
#     num_of_data = X.shape[0]
#     X_new = np.zeros((num_of_data, MAX_NUM_WAYPOINTS, X.shape[2]), dtype=np.float64)
#     # Y_new = np.zeros((num_of_data, MAX_NUM_WAYPOINTS, Y.shape[2]), dtype=np.float64)

#     for i in range(num_of_data):
#         num_of_waypoints = 0
#         for j in range(MAX_NUM_WAYPOINTS):
#             if j == 0:
#                 X_new[i, j, :] = X[i, j, :]
#                 # Y_new[i, j, :] = Y[i, j, :]
#                 num_of_waypoints += 1
#             elif j == MAX_NUM_WAYPOINTS - 1:
#                 X_new[i, j, :] = X[i, -1, :]
#                 # Y_new[i, j, :] = Y[i, -1, :]
#                 num_of_waypoints += 1
#             else:
#                 if find_significant_position_changes(X[i, :, 0:3]) or \
#                     find_significant_orientation_changes(X[i, :, 3:7]): 
#                     # find_significant_curvature_changes(X[i, :, :])[j-1]: # idk how this works
#                     X_new[i, j, :] = X[i, j, :]
#                     # Y_new[i, j, :] = Y[i, j, :]
#                     num_of_waypoints += 1
#                 else:
#                     X_new[i, j, :] = X_new[i, j-1, :]
#                     # Y_new[i, j, :] = Y_new[i, j-1, :]
#         print(f"Number of waypoints for data {i}: {num_of_waypoints}")
#     return X_new, Y


def change_trajectory_length(X):
    X_new = np.zeros((X.shape[0], MAX_TRAJ_STEPS, EMBED_DIM), dtype=np.float64) # No +1 this time
    max_index = X.shape[1]
    step_size = int(max_index / MAX_TRAJ_STEPS)
    remainder = max_index % MAX_TRAJ_STEPS
    for e_id in range(X.shape[0]):
        X_new[e_id] = np.array([np.array(X[e_id, i, 1:EMBED_DIM+1]) for i in range(0, max_index - remainder, step_size)])
    return X_new

def select_waypoints(X):
    X_new = np.zeros((X.shape[0], MAX_NUM_WAYPOINTS, EMBED_DIM), dtype=np.float64)
    for e_id in range(X.shape[0]):
        embedding = X[e_id, :, :]
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
        last_non_zero_index = all_zero_indices[0][0] - 1
        X_new[e_id, last_non_zero_index + 1:, :] = X_new[e_id, last_non_zero_index, :]

    
    return X_new


def prepare_model_input(X):
    num_experiments = X.shape[0]
    data_per_experiment = MAX_NUM_WAYPOINTS 
    new_X_num_data = data_per_experiment * num_experiments
    
    X_new = np.zeros((new_X_num_data, MAX_NUM_WAYPOINTS, EMBED_DIM), dtype=np.float64)
    Y_new = np.zeros((new_X_num_data, EMBED_DIM - 1), dtype=np.float64) # No cup type

    for e_id in range(X.shape[0]):
        for i in range(data_per_experiment):
            X_new[e_id * data_per_experiment + i, 0: int((i+1) * X.shape[1]/data_per_experiment), :] = X[e_id, 0: int((i+1) * X.shape[1]/data_per_experiment), :]
            x_index = int((i+1) * X.shape[1]/data_per_experiment)
            if x_index > X.shape[1] - 1:
                x_index = X.shape[1] - 1
            Y_new[e_id * data_per_experiment + i, :] = X[e_id, x_index, 0:EMBED_DIM-1]
    
    return X_new, Y_new


def store_only_non_spill_trajectory(X, Y):
    non_zero_indices = np.nonzero(Y)[0]
    X_new = X[non_zero_indices]
    Y_new = Y[non_zero_indices]
    return X_new, Y_new


def process_data():
    X, Y = read_from_mocap_file()
    X = change_trajectory_length(X) # removes timestamp
    X, Y = store_only_non_spill_trajectory(X, Y)
    X = transform_trajectory(X) # why is this making it worse?
    X, Y = add_equivalent_quaternions(X, Y)
    X = select_waypoints(X)
    X, Y = prepare_model_input(X)
    return X, Y

if __name__ == "__main__":
    X, Y = process_data()
    print(X[9, :, :])
    plot_X(X, 9, 0.01)
    # plot both X and Y
    