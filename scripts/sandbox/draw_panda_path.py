import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from trajectory_model.helper.read import read_panda_vectors
from trajectory_model.helper.helper import calculate_endpoint


file_name = '21-11-2023 15-44-04'
# file_name = '01-09-2023 13-58-43'
# file_name = "01-09-2023 14-09-56"

file_path = '/home/ava/projects/assets/cartesian/'+file_name+'/cartesian_positions.bin'
vectors = read_panda_vectors(file_path)
# print(vectors[0])

def get_start_end_points(cup_frame=False):
    start_points = np.empty((0, 3), np.float64)
    end_points = np.empty((0, 3), np.float64)
    for row in vectors:
        x, y, z = row[0], row[1], row[2]
        w, a, b, c = row[3], row[4], row[5], row[6]
        start_point = np.array([[x, y, z]])

        end_point = calculate_endpoint(start_point, a, b, c, w)
        start_points = np.append(start_points, start_point, axis=0)
        end_points = np.append(end_points, end_point, axis=0)
    return start_points, end_points


if __name__ == "__main__":
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