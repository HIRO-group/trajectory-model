import numpy as np

from trajectory_model.helper import find_significant_curvature_changes, \
    find_significant_orientation_changes, \
    find_significant_position_changes

from trajectory_model.constants import MAX_NUM_WAYPOINTS
from process_data import read_from_mocap_file, fix_trajectory_lenght


def select_waypoints(X, Y):
    num_of_data = X.shape[0]
    X_new = np.zeros((num_of_data, MAX_NUM_WAYPOINTS, X.shape[2]), dtype=np.float64)
    # Y_new = np.zeros((num_of_data, MAX_NUM_WAYPOINTS, Y.shape[2]), dtype=np.float64)

    for i in range(num_of_data):
        num_of_waypoints = 0
        for j in range(MAX_NUM_WAYPOINTS):
            if j == 0:
                X_new[i, j, :] = X[i, j, :]
                # Y_new[i, j, :] = Y[i, j, :]
                num_of_waypoints += 1
            elif j == MAX_NUM_WAYPOINTS - 1:
                X_new[i, j, :] = X[i, -1, :]
                # Y_new[i, j, :] = Y[i, -1, :]
                num_of_waypoints += 1
            else:
                if find_significant_position_changes(X[i, :, 0:3]) or \
                    find_significant_orientation_changes(X[i, :, 3:7]): 
                    # find_significant_curvature_changes(X[i, :, :])[j-1]: # idk how this works
                    X_new[i, j, :] = X[i, j, :]
                    # Y_new[i, j, :] = Y[i, j, :]
                    num_of_waypoints += 1
                else:
                    X_new[i, j, :] = X_new[i, j-1, :]
                    # Y_new[i, j, :] = Y_new[i, j-1, :]
        print(f"Number of waypoints for data {i}: {num_of_waypoints}")
    return X_new, Y


def process_data():
    X, Y = read_from_mocap_file()

    # X: (num_of_data, MAX_TRAJ_STEPS, EMBED_DIM), Y: (num_of_data, 1)
    X = fix_trajectory_lenght(X)
    
    # X: (num_of_data, MAX_NUM_WAYPOINTS, EMBED_DIM), 
    # Y: (num_of_data, MAX_NUM_WAYPOINTS, EMBED_DIM)
    X, Y = select_waypoints(X, Y) 
    
    return X, Y

if __name__ == "__main__":
    X, Y = process_data()