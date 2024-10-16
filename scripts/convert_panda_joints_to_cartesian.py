import numpy as np
import os
import struct
import csv
from scipy.spatial.transform import Rotation

from trajectory_model.common import perform_fk


def read_panda_vectors(filename):
    vectors = []
    with open(filename, 'rb') as f:
        num_vectors = struct.unpack('Q', f.read(8))[0]
        for _ in range(num_vectors):
            vector = [struct.unpack('d', f.read(8))[0] for _ in range(7)] 
            vectors.append(vector)            
    return vectors


def get_panda_joint_angles(file_name):
    panda_file_path =  'data/panda/joint_space/'+file_name+'/joint_waypoints.bin'
    trajectory = read_panda_vectors(panda_file_path)
    trajectory = np.array([np.array(trajectory[i]) for i in range(0, len(trajectory), 1)])
    return trajectory


def write_cartesian_to_csv(trajectory, file_name):
    directory = 'data/panda/end_effector_space/' + file_name
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(directory + '/cartesian.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in trajectory:
            writer.writerow(row)


if __name__ == "__main__":
    file_names = os.listdir('data/panda/joint_space')
    for file_name in file_names:
        joint_angles = get_panda_joint_angles(file_name)
        cartesian_trajectory = []
        for joint_angle in joint_angles:
            ee_matrix = perform_fk(joint_angle)
            ee_quat = Rotation.from_matrix(ee_matrix[:3, :3]).as_quat()

            ee_pos = ee_matrix[:3, 3]
            ee = np.array([ee_pos[0], ee_pos[1], ee_pos[2], ee_quat[0], ee_quat[1], ee_quat[2], ee_quat[3]])
            cartesian_trajectory.append(ee)
        
        cartesian_trajectory = np.array(cartesian_trajectory)
        write_cartesian_to_csv(trajectory=cartesian_trajectory, file_name=file_name)
