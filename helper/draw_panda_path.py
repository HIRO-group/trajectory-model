from read import read_vectors
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

from helper.rotate_quaternion import quaternion_to_angle_axis, rotate_quaternion
from trajectory_model.helper.helper import quat_to_euler, euler_to_quat, ctime_str

# file_name = '01-09-2023 13-42-14'
# file_name = '01-09-2023 13-58-43'
file_name = "01-09-2023 14-09-56"

file_path = '/home/ava/projects/assets/cartesian/'+file_name+'/cartesian_positions.bin'
vectors = read_vectors(file_path)

print("vectors[0]", vectors[0])
def rotate_vector(vector, rotation_matrix):
    return np.dot(rotation_matrix, vector)

def calculate_endpoint(start, a, b, c, d):
    rotation_matrix = R.from_quat([a, b, c, d]).as_matrix()
    unit_vector = np.array([0, 0, 1])
    endpoint = rotate_vector(unit_vector, rotation_matrix)
    return start + endpoint

def get_start_end_points():
    start_points = np.empty((0, 3), np.float64)
    end_points = np.empty((0, 3), np.float64)

    q_start = (vectors[0][3], vectors[0][4], vectors[0][5], vectors[0][6])
    q = (-0.0045020487159490585, -0.001336313085630536, -0.21967099606990814, -0.9755628108978271)
    angle, axis = quaternion_to_angle_axis(q, q_start)

    for row in vectors:
        x, y, z = row[0], row[1], row[2]
        a, b, c, d = row[3], row[4], row[5], row[6]

        x_rot, y_rot, z_rot, w_rot = rotate_quaternion((a, b, c, d), angle, axis)
        # a, b, c, d = x_rot, y_rot, z_rot, w_rot

        start_point = np.array([[x, y, z]])
        end_point = calculate_endpoint(start_point, a, b, c, d)
        start_points = np.append(start_points, start_point, axis=0)
        end_points = np.append(end_points, end_point, axis=0)
    
    # start_points = start_points - start_points[0]
    # end_points = end_points - start_points[0]
    return start_points, end_points


start_points, end_points = get_start_end_points()
start_points = start_points[0:5942:100, :]
end_points = end_points[0:5942:100, :]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.quiver(start_points[:, 0], start_points[:, 1], start_points[:, 2],
            end_points[:, 0], end_points[:, 1], end_points[:, 2],
            length = 0.03, normalize = True)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()