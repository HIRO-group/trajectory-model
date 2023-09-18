import numpy as np

from trajectory_model.helper.helper import find_significant_curvature_changes, \
    find_significant_orientation_changes, \
    find_significant_position_changes

from trajectory_model.spill_free.constants import MAX_TRAJ_STEPS, EMBED_DIM
from trajectory_model.informed_sampler.constants import MAX_NUM_WAYPOINTS
from process_data.process_data_SFC import read_from_mocap_file, transform_trajectory, add_equivalent_quaternions, read_from_ompl_file
from process_data.process_data_SFC import process_data as process_data_spill
from trajectory_model.helper.helper import plot_X, plot_multiple_e_ids, plot_multiple_X


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
        else:
            last_non_zero_index = all_zero_indices[0][0] - 1
            X_new[e_id, last_non_zero_index + 1:, :] = X_new[e_id, last_non_zero_index, :]
    return X_new


def prepare_model_input(X):
    num_experiments = X.shape[0]
    data_per_experiment = MAX_NUM_WAYPOINTS
    new_X_num_data = MAX_NUM_WAYPOINTS * num_experiments
    
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
    # non_zero_indices = np.nonzero(Y)[0]
    # print(non_zero_indices)
    no_spill_indices =  np.where(Y == 0)[0]
    # print(no_spill_indices)
    X_new = X[no_spill_indices]
    Y_new = Y[no_spill_indices]
    return X_new, Y_new


def convert_to_cm(X):
    X[:, :, 0:3] = np.round(X[:, :, 0:3] * 100, 0)
    return X


def process_data():
    X, Y = read_from_mocap_file()
    # X, Y = read_from_ompl_file(X, Y)

    X = change_trajectory_length(X) # removes timestamp
    X, Y = store_only_non_spill_trajectory(X, Y)
    X = transform_trajectory(X)
    X = convert_to_cm(X)
    X, Y = add_equivalent_quaternions(X, Y)
    X = select_waypoints(X)
    X, Y = prepare_model_input(X)
    
    return X, Y

if __name__ == "__main__":
    X, Y = process_data()
    # print("X[9]: ", X[9, :, 0:3])
    # plot_multiple_e_ids(X, [9, 2], 3)
    # X_s, Y_s = process_data_spill()
    # X_s[:, :, 0:3] = np.round(X_s[:, :, 0:3] * 100, 0)
    # print("X.shape: ", X.shape)
    # print("Y.shape: ", Y.shape)
    # print(X[3])
    # print(Y[3])
    Y = Y.reshape(Y.shape[0], 1, Y.shape[1])
    plot_multiple_X([X, Y], [3, 3], 1)    
    # plot_X_Y(X, Y, 0, 1)
