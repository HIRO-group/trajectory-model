from scipy.spatial.transform import Rotation as R
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def rotate_vector(vector, rotation_matrix):
    return np.dot(rotation_matrix, vector)

def quat_to_euler(quaternions):
    rot = R.from_quat(quaternions)
    rot_euler = rot.as_euler('xyz', degrees=True)
    return rot_euler

def euler_to_quat(euler_angles):
    rot = R.from_euler('xyz', euler_angles, degrees=True)
    rot_quat = rot.as_quat()
    return rot_quat

def calculate_endpoint(start, a, b, c, d):
    rotation_matrix = R.from_quat([a, b, c, d]).as_matrix()
    unit_vector = np.array([0, 0, 1])
    endpoint = rotate_vector(unit_vector, rotation_matrix)
    return start + endpoint


def get_start_and_end_points(X, e_id):
    start_points = np.empty((0, 3), np.float64)
    end_points = np.empty((0, 3), np.float64)
    
    for step in range(X.shape[1]):
        x, y, z = X[e_id, step, 0], X[e_id, step, 1], X[e_id, step, 2]
        a, b, c, d = X[e_id, step, 3], X[e_id, step, 4], X[e_id, step, 5], X[e_id, step, 6]
        all_zeros = not np.any(X[e_id, step, :])
        if all_zeros:
            break
        
        start_point = np.array([[x, y, z]])
        end_point = calculate_endpoint(start_point, a, b, c, d)
        start_points = np.append(start_points, start_point, axis=0)
        end_points = np.append(end_points, end_point, axis=0)

    return start_points, end_points


def plot_X(X, e_id, arrows_lenght, verbose=False):
    start_points, end_points = get_start_and_end_points(X, e_id)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.quiver(start_points[:, 0], start_points[:, 1], start_points[:, 2],
                end_points[:, 0], end_points[:, 1], end_points[:, 2],
                length = arrows_lenght, normalize = True)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()