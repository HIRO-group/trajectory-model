import numpy as np
from trajectory_model.helper.read import read_panda_vectors
from sandbox.constants import Tasks
from scipy.spatial.transform import Rotation
from trajectory_model.helper.fk import perform_fk

import csv
import struct
import os


def get_panda_joint_angles(file_name):
    panda_file_path =  '/home/ava/projects/assets/'+file_name+'/joint_waypoints.bin'
    trajectory = read_panda_vectors(panda_file_path)
    trajectory = np.array([np.array(trajectory[i]) for i in range(0, len(trajectory), 1)])
    return trajectory


def write_cartesian_to_file(cartesian_traj, file_name):
    if not os.path.exists('/home/ava/projects/assets/cartesian_fixed/'+file_name):
        os.makedirs('/home/ava/projects/assets/cartesian_fixed/'+file_name)

    with open('/home/ava/projects/assets/cartesian_fixed/'+file_name+'/cartesian.bin', 'wb') as f:
        f.write(struct.pack('Q', len(cartesian_traj)))
        for i in range(len(cartesian_traj)):
            for j in range(7):
                f.write(struct.pack('d', cartesian_traj[i][j]))
        


if __name__ == "__main__":
    raise Exception("This script is not meant to be run")
    for tasks in Tasks.all:
        for task in tasks:
            cartesian_traj = []
            joint_angles = get_panda_joint_angles(task)
            for joint_angle in joint_angles:
                ee_matrix = perform_fk(joint_angle)
                ee_quat = Rotation.from_matrix(ee_matrix[:3, :3]).as_quat()

                ee_pos = ee_matrix[:3, 3]
                ee = np.array([ee_pos[0], ee_pos[1], ee_pos[2], ee_quat[0], ee_quat[1], ee_quat[2], ee_quat[3]])
                cartesian_traj.append(ee)

            cartesian_traj = np.array(cartesian_traj)
            write_cartesian_to_file(cartesian_traj, task)
