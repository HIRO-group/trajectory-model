from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D


def rotate_vector(vector, rotation_matrix):
    return np.dot(rotation_matrix, vector)


def quat_to_euler(quaternions, degrees=True):
    rot = R.from_quat(quaternions)
    rot_euler = rot.as_euler('xyz', degrees=degrees)
    return rot_euler


def euler_to_quat(euler_angles, degrees=True):
    rot = R.from_euler('xyz', euler_angles, degrees=degrees)
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
        a, b, c, d = X[e_id, step, 3], X[e_id, step,
                                         4], X[e_id, step, 5], X[e_id, step, 6]
        all_zeros = not np.any(X[e_id, step, :])
        if all_zeros:
            print("All zeros! at step: ", step)
            break
        
        if x > 100:
            continue

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

    for idx, e_id in enumerate(e_ids):
        start_points, end_points = get_start_and_end_points(X, e_id)
        if idx == len(e_ids) - 1:
            ax.quiver(start_points[:, 0], start_points[:, 1], start_points[:, 2],
                  end_points[:, 0], end_points[:, 1], end_points[:, 2],
                  length=arrows_lenght, normalize=True, color=colors[color_id], label='SFS')
        else:
            ax.quiver(start_points[:, 0], start_points[:, 1], start_points[:, 2],
                  end_points[:, 0], end_points[:, 1], end_points[:, 2],
                  length=arrows_lenght, normalize=True, color=colors[color_id])

        color_id += 0

    # num_arrows = 100
    # directions = np.random.uniform(0, 2*np.pi, (num_arrows, 2))
    # phi = directions[:, 0]
    # theta = directions[:, 1]
    # lengths = np.random.uniform(0.01, 0.05, num_arrows)
    # x = lengths * np.sin(theta) * np.cos(phi)
    # y = lengths * np.sin(theta) * np.sin(phi)
    # z = lengths * np.cos(theta)

    x_min = -0.03
    y_min = -0.4
    z_min = -0.09
    x_max = 0.54
    y_max = 0.84
    z_max = 0.64

    num_arrows = 80

    # Generate random uniform points within the specified boundary
    x = np.random.uniform(x_min, x_max, num_arrows)
    y = np.random.uniform(y_min, y_max, num_arrows)
    z = np.random.uniform(z_min, z_max, num_arrows)

    directions = np.random.uniform(0, 2 * np.pi, num_arrows)
    inclinations = np.random.uniform(0, np.pi, num_arrows)  # Adjust for non-zero z-components

    # Calculate arrow components (x, y, z) based on directions
    arrow_length = 0.05 # Adjust the arrow length as needed
    u = arrow_length * np.sin(directions) * np.sin(inclinations)
    v = arrow_length * np.cos(directions) * np.sin(inclinations)
    w = arrow_length * np.cos(inclinations)

    ax.quiver(x, y, z, u, v, w, length=arrow_length, normalize=True, color='b', label='Uniform')


    # ax.set_xlim([x_min, x_max])
    # ax.set_ylim([y_min, y_max])
    # ax.set_zlim([z_min, z_max])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()


    plt.savefig(f'plots/trajectory_comparison.png', dpi=300)
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
            arrows_lenght += 1
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



def plot_loss_function(history):
    dt = datetime.now()
    now = dt.strftime("%H:%M:%S")

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'plots/loss/loss_{now}.png')
    plt.show()


def plot_prediction_vs_real(predicted, real):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    distance = np.linalg.norm(predicted - real)
    ax.scatter(predicted[0], predicted[1], predicted[2], color='red', label='Predicted')
    ax.scatter(real[0], real[1], real[2], color='blue', label='Real')
    ax.text((predicted[0] + real[0]) / 2, (predicted[1] + real[1]) / 2, (predicted[2] + real[2]) / 2,
        f'Distance: {distance:.2f}', color='black')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Scatter Plot')
    ax.legend()
    plt.show()



def plot_xyz(X_xyz, Y_xyz, e_id):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for waypoint in X_xyz[e_id]:
        x, y, z = waypoint[0], waypoint[1], waypoint[2]
        ax.scatter(x, y, z, color='red', label='X[e_id]')

    x, y, z = Y_xyz[e_id][0], Y_xyz[e_id][1], Y_xyz[e_id][2]
    ax.scatter(x, y, z, color='blue', label='Y[e_id]')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


def ctime_str():
    # dt = datetime.now()
    # now = dt.strftime("%H:%M:%S")
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
