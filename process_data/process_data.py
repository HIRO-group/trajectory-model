# process data for spill free model
import csv
import numpy as np
from trajectory_model.spill_free.constants import MAX_TRAJ_STEPS, EMBED_DIM, FINAL_MOCAP_RAW_DATA_DIR, FINAL_OMPL_RAW_DATA_DIR
from trajectory_model.helper import plot_X

def read_from_ompl_file(X, Y, data_dir=FINAL_OMPL_RAW_DATA_DIR):
    unsafe_indexes = [1, 2, 3, 4, 5, 6, 7, 8]
    for id in unsafe_indexes:
        path = f'{data_dir}/{id}_spill'
        X_t = np.loadtxt(path, delimiter=',')
        X_t = X_t.reshape(1, MAX_TRAJ_STEPS, EMBED_DIM)
        X = np.append(X, X_t, axis=0)
        Y = np.append(Y, np.array([[1]]), axis=0)
    return X, Y
    

def read_from_mocap_file(data_dir=FINAL_MOCAP_RAW_DATA_DIR):
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


def fix_trajectory_lenght(X):
    X_new = np.zeros((X.shape[0], MAX_TRAJ_STEPS, EMBED_DIM), dtype=np.float64) # No +1 this time
    max_index = X.shape[1]
    step_size = int(max_index / MAX_TRAJ_STEPS)
    remainder = max_index % MAX_TRAJ_STEPS
    for e_id in range(X.shape[0]):
        X_new[e_id] = np.array([np.array(X[e_id, i, 1:EMBED_DIM+1]) for i in range(0, max_index - remainder, step_size)])

    # This copies the last non-zero value to the rest of the trajectory
    for e_id in range(X_new.shape[0]):
        embedding = X_new[e_id, :, :] 
        all_zero_indices = np.where(np.all(embedding == 0, axis=1))
        if len(all_zero_indices[0]) == 0:
            continue
        first_zero_index =  all_zero_indices[0][0]
        
        if first_zero_index == 0:
            X_new[e_id, :, :] = X_new[e_id-1, :, :]
        else:
            last_non_zero_index = first_zero_index - 1
            X_new[e_id, first_zero_index:, :] = X_new[e_id, last_non_zero_index, :]
    return X_new

def add_partial_trajectory(X, Y):
    num_experiments = X.shape[0]
    data_per_experiment = 4
    new_X_num_data = data_per_experiment * num_experiments
    X_new = np.zeros((new_X_num_data, X.shape[1], EMBED_DIM), dtype=np.float64)
    Y_new = np.zeros((new_X_num_data, 1), dtype=np.float64)

    for e_id in range(X.shape[0]):
        for i in range(data_per_experiment):
            Y_new[e_id * data_per_experiment + i, :] = Y[e_id]
            X_new[e_id * data_per_experiment + i, 0: int((i+1) * X.shape[1]/data_per_experiment), :] = X[e_id, 0: int((i+1) * X.shape[1]/data_per_experiment), :]
            # fill the rest with the last value
            X_new[e_id * data_per_experiment + i, int((i+1) * X.shape[1]/data_per_experiment):, :] = X[e_id, int((i+1) * X.shape[1]/data_per_experiment)-1, :]
    return X_new, Y_new


def transform_trajectory(X):
    for e_id in range(X.shape[0]):
        xyz = X[e_id, :, 0:3]
        xyz = xyz - X[e_id, 0, 0:3]
        X[e_id, :, 0:3] = xyz
    return X


def add_equivalent_quaternions(X, Y):
    X_new_shape_0 = 2 * X.shape[0] # q = -q (quaternion)
    X_new_shape_1 = X.shape[1]
    X_new = np.zeros((X_new_shape_0, X_new_shape_1, EMBED_DIM), dtype=np.float64)
    Y_new = np.zeros((X_new_shape_0, 1), dtype=np.float64)

    for e_id in range(0, X.shape[0]):
        X_new[2 * e_id] = X[e_id]
        X_new[2 * e_id + 1] = X[e_id]
        X_new[2 * e_id + 1, :, 3:7] = -X[e_id, :, 3:7]
        Y_new[2 * e_id] = Y[e_id]
        Y_new[2 * e_id + 1] = Y[e_id]
    return X_new, Y_new


def process_data():
    X, Y = read_from_mocap_file()
    X = fix_trajectory_lenght(X) 
    # X, Y = read_from_ompl_file(X, Y)
    X = transform_trajectory(X)
    X, Y = add_equivalent_quaternions(X, Y)
    # X, Y = add_partial_trajectory(X, Y)
    return X, Y


# if __name__ == "__main__":
X, Y = process_data()
# print(X[0, 0, :])
# print(X[1, 0, :])
plot_X(X, 0, 1)
print("here: ", X[0, 0, :])
