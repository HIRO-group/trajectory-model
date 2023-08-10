from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt


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
    # print("a, b, c, d: ", a, b, c, d)
    rotation_matrix = R.from_quat([a, b, c, d]).as_matrix()
    unit_vector = np.array([0, 0, 1])
    endpoint = rotate_vector(unit_vector, rotation_matrix)
    return start + endpoint


def get_start_and_end_points(X, e_id):
    start_points = np.empty((0, 3), np.float64)
    end_points = np.empty((0, 3), np.float64)

    for step in range(X.shape[1]):
        x, y, z = X[e_id, step, 0], X[e_id, step, 1], X[e_id, step, 2]
        a, b, c, d = X[e_id, step, 3], X[e_id, step,
                                         4], X[e_id, step, 5], X[e_id, step, 6]
        all_zeros = not np.any(X[e_id, step, :])
        if all_zeros:
            print("All zeros! at step: ", step)
            break

        start_point = np.array([[x, y, z]])
        end_point = calculate_endpoint(start_point, a, b, c, d)
        start_points = np.append(start_points, start_point, axis=0)
        end_points = np.append(end_points, end_point, axis=0)

    return start_points, end_points



def plot_multiple_e_ids(X, e_ids, arrows_lenght, verbose=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b', 'b']
    color_id = 0

    for e_id in e_ids:
        start_points, end_points = get_start_and_end_points(X, e_id)
        ax.quiver(start_points[:, 0], start_points[:, 1], start_points[:, 2],
                  end_points[:, 0], end_points[:, 1], end_points[:, 2],
                  length=arrows_lenght, normalize=True, color=colors[color_id])
        color_id += 1

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

def plot_multiple_X(Xs, e_ids, arrows_lenght, verbose=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['b', 'r', 'g']
    color_id = 0
    
    for X in Xs:
        if color_id == 0:
            e_id = e_ids[0]
        else:
            e_id = e_ids[1]
            arrows_lenght += 0.03
        
        print("Xs.shape: ", X.shape)
        print("e_id: ", e_id)
        print("color: ", colors[color_id])
        

        start_points, end_points = get_start_and_end_points(X, e_id)

        ax.quiver(start_points[:, 0], start_points[:, 1], start_points[:, 2],
                end_points[:, 0], end_points[:, 1], end_points[:, 2],
                length=arrows_lenght, normalize=True, color=colors[color_id])
        color_id += 1

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


def plot_X(X, e_id, arrows_lenght, verbose=False):
    start_points, end_points = get_start_and_end_points(X, e_id)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.quiver(start_points[:, 0], start_points[:, 1], start_points[:, 2],
              end_points[:, 0], end_points[:, 1], end_points[:, 2],
              length=arrows_lenght, normalize=True)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def find_significant_position_changes(trajectory, threshold_distance=0.1):
    significant_changes = []
    for i in range(1, len(trajectory)):
        distance = calculate_distance(trajectory[i-1], trajectory[i])
        if distance > threshold_distance:
            significant_changes.append(i)
    return significant_changes


def calculate_angle(q1, q2):
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    return np.arccos(2 * np.dot(r1.as_quat(), r2.as_quat())**2 - 1)

def find_significant_orientation_changes(orientations, threshold_angle=5):
    significant_changes = []
    for i in range(1, len(orientations)):
        angle = calculate_angle(orientations[i-1], orientations[i])
        if angle > threshold_angle:
            significant_changes.append(i)
    return significant_changes


def calculate_curvature(p1, p2, p3):
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p2)
    cross_product = np.cross(v1, v2)
    return np.linalg.norm(cross_product) / np.linalg.norm(v1)

def find_significant_curvature_changes(trajectory, threshold_curvature=0.1):
    significant_changes = []
    for i in range(1, len(trajectory) - 1):
        curvature = calculate_curvature(trajectory[i-1], trajectory[i], trajectory[i+1])
        if curvature > threshold_curvature:
            significant_changes.append(i)
    return significant_changes
