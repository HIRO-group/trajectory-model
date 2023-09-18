import pandas as pd
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

def rotate_vector(vector, rotation_matrix):
    return np.dot(rotation_matrix, vector)

def calculate_endpoint(start, a, b, c, d):
    rotation_matrix = R.from_quat([a, b, c, d]).as_matrix()
    unit_vector = np.array([0, 0, 1])
    endpoint = rotate_vector(unit_vector, rotation_matrix)
    return start + endpoint

def get_xyzs(accelerations):
    positions_over_time = []
    dt = 0.017
    velocity = [0, 0, 0]
    position = [0, 0, 0]

    for i in range(1, len(accelerations)):
        for axis in range(3):
            velocity[axis] += 0.5 * (accelerations[i - 1][axis] + accelerations[i][axis]) * dt

        for axis in range(3):
            position[axis] += velocity[axis] * dt

        positions_over_time.append(position.copy())
    
    return positions_over_time

def load_data():
    X, Y = [], []
    file_paths_nospill_spill = \
                    [("Big/big-full-nospill", "Big/big-full-spill"),
                    ("Small/small-full-nospill", "Small/small-full-spill"),
                    ("Small/small-notfull-nospill", "Small/small-notfull-spill")]
    
    directory_path = "data/Phone/Small/small-full-nospill"
    file_names = os.listdir(directory_path)
    file_path = directory_path + "/" + "2023-09-03_17_49_36_my_iOS_device.json"
    print("file_path: ", file_path)

    quaternions = []
    accelerations = []

    with open(file_path, 'r') as json_file:
        datas = json.load(json_file)
        for data in datas:
            quaternion = (float(data["motionQuaternionX"]), float(data["motionQuaternionY"]),
                        float(data["motionQuaternionZ"]), float(data["motionQuaternionW"]))
            quaternions.append(quaternion)
            
            acceleration = (float(data["motionUserAccelerationX"]), float(data["motionUserAccelerationY"]),
                            float(data["motionUserAccelerationZ"]))
            accelerations.append(acceleration)

    xyzs = get_xyzs(accelerations)

    start_points = np.empty((0, 3), np.float64)
    end_points = np.empty((0, 3), np.float64)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for idx, xyz in enumerate(xyzs):
        x, y, z = xyz[0], xyz[1], xyz[2]
        a, b, c, d = quaternions[idx][0], quaternions[idx][1], quaternions[idx][2], quaternions[idx][3]
        
        start_point = np.array([[x, y, z]])
        end_point = calculate_endpoint(start_point, a, b, c, d)

        start_points = np.append(start_points, start_point, axis=0)
        end_points = np.append(end_points, end_point, axis=0)

    
    start_points = start_points[::5]
    end_points = end_points[::5]
    
    ax.quiver(start_points[:, 0], start_points[:, 1], start_points[:, 2],
                end_points[:, 0], end_points[:, 1], end_points[:, 2],
                length=1, normalize=True)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


    
    return X, Y

if __name__ == "__main__":
    X, Y = load_data()