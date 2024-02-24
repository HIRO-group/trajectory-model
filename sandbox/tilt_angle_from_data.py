import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from trajectory_model.helper.read import read_panda_vectors
from trajectory_model.helper.read import read_panda_vectors

from sandbox.constants import Tasks

def get_panda_traj(file_name):
    panda_file_path =  '/home/ava/projects/assets/cartesian/'+file_name+'/cartesian_positions.bin'
    trajectory = read_panda_vectors(panda_file_path)
    trajectory = np.array([np.array(trajectory[i]) for i in range(0, len(trajectory), 1)])
    trajectory = trajectory[np.newaxis, :, :]
    return trajectory


def transform_ee_to_cup_frame(panda_traj):
    R_be0 = R.from_quat(panda_traj[0, 0, 3:7])  
    t_be0 = panda_traj[0, 0, :3]
    T_be0 = np.eye(4)
    T_be0[:3, :3] = R_be0.as_matrix()
    T_be0[:3, 3] = t_be0

    T_bc0 = np.eye(4)
    T_bc0[:3, 3] = panda_traj[0, 0, :3]
    T_ce = np.linalg.inv(T_bc0) @ T_be0
    T_ec = np.linalg.inv(T_ce)

    tilt_angles = []
    euler_angles = []
    for step in range(0, panda_traj.shape[1]):
        R_be = R.from_quat(panda_traj[0, step, 3:7])  
        t_be = panda_traj[0, step, :3]
        T_be = np.eye(4)
        T_be[:3, :3] = R_be.as_matrix()
        T_be[:3, 3] = t_be

        T_bc = T_be @ T_ec

        euler_bc = R.from_matrix(T_bc[:3, :3]).as_euler('xyz', degrees=True)
        euler_angles.append(euler_bc)

        
        R_c = T_bc[:3, :3]
        z_axis = R_c @ np.array([0, 0, 1])
        z_axis = z_axis / np.linalg.norm(z_axis)
        ref_z = np.array([0, 0, 1])
        dot = np.dot(z_axis, ref_z)
        angle = np.arccos(dot)
        tilt_angle = np.degrees(angle)
        tilt_angles.append(tilt_angle)
    

    euler_angles = np.array(euler_angles)
    max_roll = np.max(euler_angles[:, 0])
    max_pitch = np.max(euler_angles[:, 1])
    min_roll = np.min(euler_angles[:, 0])
    min_pitch = np.min(euler_angles[:, 1])

    # print("Roll (max, min): ", (max_roll, min_roll),)
    # print("Pitch (max, min): ", (max_pitch, min_pitch))
    print("Tilt angles (max, min): ", (max(tilt_angles), min(tilt_angles)))
    print("--------")


if __name__ == "__main__":
    for task in Tasks.task_6:
        file_name = task
        panda_traj = get_panda_traj(file_name)
        transform_ee_to_cup_frame(panda_traj)