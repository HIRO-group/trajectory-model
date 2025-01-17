import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from trajectory_model.process_data.data_processor import read_panda_trajectory
from trajectory_model.process_data.panda_helper import process_panda_to_model_input
from trajectory_model.SFC.constants import BLANK_VAL


def rotate_vector(vector, rotation_matrix):
    return np.dot(rotation_matrix, vector)

def calculate_endpoint(start, a, b, c, d):
    rotation_matrix = R.from_quat([a, b, c, d]).as_matrix()
    unit_vector = np.array([0, 0, 1])
    endpoint = rotate_vector(unit_vector, rotation_matrix)
    return start + endpoint


def get_start_end_points(trajectory):
    start_points = np.empty((0, 3), np.float64)
    end_points = np.empty((0, 3), np.float64)
    for row in trajectory:
        x, y, z = row[0], row[1], row[2]
        a, b, c, w = row[3], row[4], row[5], row[6]
        start_point = np.array([[x, y, z]])
        end_point = calculate_endpoint(start_point, a, b, c, w)
        start_points = np.append(start_points, start_point, axis=0)
        end_points = np.append(end_points, end_point, axis=0)
    return start_points, end_points


def plot_quivers(start_points, end_points, quiver_length=0.02):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.quiver(start_points[:, 0], start_points[:, 1], start_points[:, 2],
                end_points[:, 0], end_points[:, 1], end_points[:, 2],
                length = quiver_length, normalize = True)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


if __name__ == "__main__":
    file_address = 'data/panda/end_effector_space/01-09-2023 13-42-14/cartesian.csv'
    
    in_cup_frame = True
    quiver_length = 0.02 # Manually set the length of the quiver arrows

    trajectory = read_panda_trajectory(file_address)
    trajectory = np.array([np.array(trajectory[i]) for i in range(0, len(trajectory))])

    if in_cup_frame:
        trajectory = process_panda_to_model_input(trajectory)
        trajectory = [tr for tr in trajectory if tr[0] < BLANK_VAL]
    start_points, end_points = get_start_end_points(trajectory)
    plot_quivers(start_points=start_points, end_points=end_points, quiver_length=quiver_length)
