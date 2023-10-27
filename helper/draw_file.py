import numpy as np
import csv

from trajectory_model.helper.helper import plot_X, plot_multiple_e_ids, plot_multiple_X

def read_file(file_path):
    X = np.zeros((1, 1000, 7), dtype=np.float64)
    trajectory_index = 0
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            keys = list(row.values())
            y, x, z = np.float64(keys[1]), np.float64(keys[2]), np.float64(keys[3])
            a, b, c, d = np.float64(keys[4]), np.float64(keys[5]), np.float64(keys[6]), np.float64(keys[7])
            embedding = np.array([[x, y, z, a, b, c, d]])
            X[0, trajectory_index, :] = embedding
            trajectory_index += 1
    return X


PREFIX = "/home/ava/projects/trajectory-model/data/"
file = PREFIX + "mocap_new/big/full/spill-free/2023-09-08 20:24:51.csv"

X = read_file(file)

plot_X(X, 0, 0.05)