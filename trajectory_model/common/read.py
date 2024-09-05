import numpy as np
import csv
import struct
# from trajectory_model.helper.helper import quat_to_euler
# from trajectory_model.helper.rotate_quaternion import rotate_new_cup_to_match_orientation

def read_panda_vectors(filename):
    vectors = []
    with open(filename, 'rb') as f:
        num_vectors = struct.unpack('Q', f.read(8))[0]
        for _ in range(num_vectors):
            vector = [struct.unpack('d', f.read(8))[0] for _ in range(7)] 
            vectors.append(vector)            
    return vectors


def read_mocap_file(file_path):
    X = np.zeros((1, 2000, 7), dtype=np.float64)
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

    X = X[:, 0:2000:10, :]
    X[:, :, 0:3] = X[:, :, 0:3] - X[:, 0, 0:3]
    
    return X