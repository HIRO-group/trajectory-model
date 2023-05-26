import csv
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial.transform import Rotation as R
from itertools import islice

from constants import DATA_DIR

def rotate_vector(vector, rotation_matrix):
    return np.dot(rotation_matrix, vector)

def calculate_endpoint(start, a, b, c, d):
    rotation_matrix = R.from_quat([a, b, c, d]).as_matrix()
    unit_vector = np.array([0, 0, 1])
    endpoint = rotate_vector(unit_vector, rotation_matrix)
    return start + endpoint

def read_experiment(experiment_id, num_samples=100):
    start_points = np.empty((0, 3), np.float64)
    end_points = np.empty((0, 3), np.float64)
    with open(DATA_DIR, mode ='r')as file:
        reader = csv.DictReader(file)
        for row in reader:
            keys = list(row.values())
            e_id, timestamp = int(keys[0]), keys[1]
            x, y, z = np.float64(keys[2]), np.float64(keys[3]), np.float64(keys[4]),
            a, b, c, d = np.float64(keys[5]), np.float64(keys[6]), np.float64(keys[7]), np.float64(keys[8])

            if e_id == experiment_id:
                start_point = np.array([[x, y, z]])
                end_point = calculate_endpoint(start_point, a, b, c, d)
                start_points = np.append(start_points, start_point, axis=0)
                end_points = np.append(end_points, end_point, axis=0)

            if e_id > experiment_id or len(start_points) > num_samples:
                break
    return start_points, end_points


def visualize_data_with_orientation(experiment_id, start_points, end_points, arrows_lenght=0.001):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(start_points[:, 0], start_points[:, 1], start_points[:, 2],
                end_points[:, 0], end_points[:, 1], end_points[:, 2],
                length = arrows_lenght, normalize = True)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.savefig('plots/experiment_{}.png'.format(experiment_id))
    plt.show()


def visualize_data(x_list, y_list, z_list):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_list, y_list, z_list)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


experiment_id = 1
starts, ends = read_experiment(experiment_id=experiment_id, num_samples=100)
visualize_data_with_orientation(experiment_id, starts, ends, arrows_lenght=0.001)
