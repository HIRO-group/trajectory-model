import csv
import numpy as np

from constants import DATA_DIR, MAX_TRAJ_STEPS, EMBED_DIM

# def read_experiment(experiment_id, data_dir=DATA_DIR):
    
#     with open(data_dir, mode ='r')as file:
#         reader = csv.DictReader(file)
#         for row in reader:
#             keys = list(row.values())
#             e_id, timestamp = int(keys[0]), keys[1]
#             x, y, z = np.float64(keys[2]), np.float64(keys[3]), np.float64(keys[4]),
#             a, b, c, d = np.float64(keys[5]), np.float64(keys[6]), np.float64(keys[7]), np.float64(keys[8])
#             if e_id > experiment_id:
#                 break

def process_data(data_dir=DATA_DIR):
    exclude_indexes_list = [4, 6, 11, 12, 13, 14, 15, 16, 22, 23, 25, 31]
    safe_data_index_range = (1, 21)
    unsafe_data_index_range = (21, 35)
    
    num_data = unsafe_data_index_range[1] - safe_data_index_range[0] + 1 - len(exclude_indexes_list)
    x = np.zeros((num_data, MAX_TRAJ_STEPS, EMBED_DIM), dtype=np.float64)

    for experiment_id in safe_data_index_range:

        with open(data_dir, mode ='r')as file:
            reader = csv.DictReader(file)
            for row in reader:
                keys = list(row.values())
                e_id, timestamp = int(keys[0]), keys[1]
                x, y, z = np.float64(keys[2]), np.float64(keys[3]), np.float64(keys[4]),
                a, b, c, d = np.float64(keys[5]), np.float64(keys[6]), np.float64(keys[7]), np.float64(keys[8])
                # should convert timestamp to 
                x[e_id, timestamp, :] = np.array([[x, y, z, a, b, c, d, 1]]) # cup type is 1


X, Y = process_data()