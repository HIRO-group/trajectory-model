from datetime import datetime
import csv
import numpy as np

from trajectory_model.constants import EMBED_DIM, FINAL_RAW_DATA_DIR, FINAL_PROCESSED_DATA_DIR_PREFIX
from trajectory_model.helper import calculate_endpoint, plot_X

def read_from_file(data_dir=FINAL_RAW_DATA_DIR):
    exclude_indexes_list = [4, 6, 11, 12, 13, 14, 15, 16, 22, 23, 25, 31]
    safe_data_index_range = (1, 21)
    unsafe_data_index_range = (21, 35)
    num_data = unsafe_data_index_range[1] - safe_data_index_range[0] + 1 - len(exclude_indexes_list)
    
    X = np.zeros((num_data, 1000, EMBED_DIM + 1), dtype=np.float64) # + 1 for timestamp
    Y = np.zeros((num_data, 1))
    seen_experiments = []
    
    with open(data_dir, mode ='r')as file:
        reader = csv.DictReader(file)
        correct_e_id = 0
        
        for row in reader:
            keys = list(row.values())
            
            e_id, timestamp = int(keys[0]), np.float64(keys[1])
            if e_id in exclude_indexes_list:
                if e_id not in seen_experiments:
                    correct_e_id += 1

            elif e_id not in exclude_indexes_list:
                if e_id not in seen_experiments:
                    traj_index = 0
                else:
                    traj_index += 1

            seen_experiments.append(e_id)

            if e_id in exclude_indexes_list:
                continue

            x, y, z = np.float64(keys[2]), np.float64(keys[3]), np.float64(keys[4]),
            a, b, c, d = np.float64(keys[5]), np.float64(keys[6]), np.float64(keys[7]), np.float64(keys[8])
            
            X[e_id-1-correct_e_id, traj_index, :] = np.array([[timestamp, x, y, z, a, b, c, d, 1]]) # cup type is 1
            
            if e_id in range(safe_data_index_range[0], safe_data_index_range[1]):
                Y[e_id-1-correct_e_id] = 0
            else:
                Y[e_id-1-correct_e_id] = 1
    return X, Y


def get_biggest_trajectory(X):
    max_T_e_id = 0
    max_zero_index = 0
    for e_id in range(X.shape[0]):
        first_zero_index = np.argmin(X[e_id, :, 0])
        if first_zero_index > max_zero_index:
            max_zero_index = first_zero_index
            max_T_e_id = e_id
    # t2 = datetime.fromtimestamp(X[max_T_e_id, max_zero_index - 1, 0])
    # t1 = datetime.fromtimestamp(X[max_T_e_id, 0, 0])
    return max_zero_index

def fix_trajectory_lenght(X):
    max_zero_index = get_biggest_trajectory(X)
    frame_rate = 120
    dt = int(frame_rate/10)
    traj_lenght = int(max_zero_index/dt) + 1
    X_new = np.zeros((X.shape[0], traj_lenght, EMBED_DIM), dtype=np.float64) # No +1 this time
    X_new[:, :, :] = X[:, 0:max_zero_index:dt, 0:EMBED_DIM]
    return X_new

def write_to_csv(X, Y, data_dir_prefix = FINAL_PROCESSED_DATA_DIR_PREFIX):
    # write to csv
    # with open(data_dir, mode ='w')as file:
    # np.savetxt(f'{data_dir_prefix}_X.csv', X, delimiter=",")
    # np.savetxt(f'{data_dir_prefix}_Y.csv', Y, delimiter=",")
    pass



def translate(body, translation):
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = translation
    translated_body = np.dot(body, translation_matrix.T)
    return translated_body

def translate_trajectory(X):
    for e_id in range(X.shape[0]):
        trajectory = X[e_id, :, 1:4]
        start_point = trajectory[0, :]
        translation = -start_point
        rigid_bodies = np.hstack((trajectory, np.ones((trajectory.shape[0], 1))))
        trajectory = translate(rigid_bodies, translation)
        X[e_id, :, 0:3] = trajectory[:, 0:3]
    return X


def rotate_trajectory(X):
    return X

def add_partial_trajectory(X, Y):
    num_experiments = X.shape[0]
    traj_lenght = X.shape[1]
    data_per_experiment = 4

    new_X_num_data = data_per_experiment * num_experiments
    
    X_new = np.zeros((new_X_num_data, traj_lenght, EMBED_DIM), dtype=np.float64)
    Y_new = np.zeros((new_X_num_data, 1), dtype=np.float64)
    
    for e_id in range(X.shape[0]):
        for i in range(data_per_experiment):
            X_new[e_id * data_per_experiment + i , i * data_per_experiment:(i+1) * data_per_experiment, :] = \
                 X[e_id, i * data_per_experiment:(i+1) * data_per_experiment, :]
            Y_new[e_id * data_per_experiment + i, :] = Y[e_id]

    return X_new, Y_new


def process_data(data_dir=FINAL_RAW_DATA_DIR):
    X, Y = read_from_file()
    X = fix_trajectory_lenght(X)
    X = translate_trajectory(X)
    X, Y = add_partial_trajectory(X, Y)
    X = rotate_trajectory(X)
    return X, Y


if __name__ == "__main__":
    X, Y = process_data()
    write_to_csv(X, Y)