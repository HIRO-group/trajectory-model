import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from trajectory_model.classifier_predict_func_api import process_panda_to_model_input
from trajectory_model.helper.read import read_panda_vectors
from trajectory_model.helper.helper import quat_to_euler
from trajectory_model.helper.rotate_quaternion import rotate_panda_to_match_orientation
from trajectory_model.classifier_predict_func_api import process_panda_to_model_input
from trajectory_model.helper.read import read_panda_vectors
from trajectory_model.helper.helper import quat_to_euler, get_start_and_end_points

from sandbox.constants import Tasks


def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_rot_panda_traj(file_name):
    panda_file_path =  '/home/ava/projects/assets/cartesian/'+file_name+'/cartesian_positions.bin'
    
    trajectory = read_panda_vectors(panda_file_path)

    trajectory = np.array([np.array(trajectory[i]) for i in range(0, len(trajectory), 1)])
    trajectory = rotate_panda_to_match_orientation(trajectory)

    trajectory = trajectory[np.newaxis, :, :]
    return trajectory


def get_tilt_angles(panda_traj):
    start_points, end_points = get_start_and_end_points(panda_traj, 0)    
    vectors = end_points - start_points
    vertical_vector = np.array([0, 0, 1])
    tilt_angles = []
    for vector in vectors:
        angle = angle_between(vector, vertical_vector)
        tilt_angles.append(np.degrees(angle))
    print("max tilt angle: ", max(tilt_angles))
    print("min tilt angle: ", min(tilt_angles))
    print("--------")


def get_max_roll_pitch(panda_traj):
    quatertions = panda_traj[0, :, 3:7]
    euler_angles = []
    for q in quatertions:
        euler_angles.append(quat_to_euler(q))
    euler_angles = np.array(euler_angles)
    roll = euler_angles[:, 0]
    pitch = euler_angles[:, 1]
    max_roll = max(roll)
    max_pitch = max(pitch)
    # print("max roll: ", max_roll, ", max pitch: ", max_pitch)
    # print("max pitch: ", max_pitch)
    # print max roll, pitch, and min roll, pitch
    print("max roll: ", max(roll), ", max pitch: ", max(pitch))
    print("min roll: ", min(roll), ", min pitch: ", min(pitch))
    print("-----")


def rotation_matrixes(panda_traj):
    R_bc = R.from_quat([0, 0, 0, 1]).as_matrix() # cup in base frame in t=0

    euler_angles = []
    tilt_angles = []
    for step in range(1, panda_traj.shape[1]):
        R_be1 = R.from_quat(panda_traj[0, step-1, 3:7]).as_matrix()
        R_be2 = R.from_quat(panda_traj[0, step, 3:7]).as_matrix()

        R_e1e2 = np.dot(np.linalg.inv(R_be1), R_be2)
        R_bc = R_be2 @ np.linalg.inv(R_be1) @ R_bc

        # R_bc1 = R_be1 * Inv(R_be0) * R_bc0

        # euler_bc_new = R_bc_new.as_euler('xyz', degrees=True)
        # euler_angles.append(euler_bc_new)

        unit_vector = np.array([0, 0, 1])
        end_point = np.dot(R_bc, unit_vector)
        angle = angle_between(end_point, unit_vector)
        tilt_angles.append(np.degrees(angle))


    euler_angles = np.array(euler_angles)
    print("(min, max) tilt angles: ", min(tilt_angles), max(tilt_angles))
    # print("max roll: ", max(euler_angles[:, 0]), ", max pitch: ", max(euler_angles[:, 1]))
    # print("min roll: ", min(euler_angles[:, 0]), ", min pitch: ", min(euler_angles[:, 1]))
    # print("-----")


def transform_matrices(panda_traj):
    T_bc = np.eye(4) 
    T_bc[:3, 3] = panda_traj[0, 0, :3]
    T_bc[2, 3] = 0.35

    euler_angles = []
    tilt_angles = []
    for step in range(1, panda_traj.shape[1]):

        R_be1 = R.from_quat(panda_traj[0, step-1, 3:7])  
        t_be1 = panda_traj[0, step-1, :3]
        T_be1 = np.eye(4)
        T_be1[:3, :3] = R_be1.as_matrix()
        T_be1[:3, 3] = t_be1
        
        R_be2 = R.from_quat(panda_traj[0, step, 3:7])
        t_be2 = panda_traj[0, step, :3]
        T_be2 = np.eye(4)
        T_be2[:3, :3] = R_be2.as_matrix() 
        T_be2[:3, 3] = t_be2
        
        T_e1e2 = np.linalg.inv(T_be1) @ T_be2
        T_bc = T_e1e2 @ T_bc


        euler_bc = R.from_matrix(T_bc[:3, :3]).as_euler('xyz', degrees=True)
        euler_angles.append(euler_bc)


        z_axis = T_bc[:3, :3] @ np.array([0, 0, 1])
        z_axis = z_axis / np.linalg.norm(z_axis)
        ref_z = np.array([0, 0, 1])
        dot = np.dot(z_axis, ref_z)
        angle = np.arccos(dot)
        tilt_angles.append(np.degrees(angle))


        # tilt_angle = np.degrees(angle_between(T_bc[:3, 2], np.array([0, 0, 1])))
        # tilt_angles.append(tilt_angle)

        # end_effector_poses.append(T_be2) 
        # container_poses.append(T_bc)

    euler_angles = np.array(euler_angles)
    # print("max roll: ", max(euler_angles[:, 0]), ", max pitch: ", max(euler_angles[:, 1]))
    # print("min roll: ", min(euler_angles[:, 0]), ", min pitch: ", min(euler_angles[:, 1]))
    print("(min, max) tilt angles: ", min(tilt_angles), max(tilt_angles))



def alex_method(panda_traj):
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

    print("Roll (max, min): ", (max_roll, min_roll),)
    print("Pitch (max, min): ", (max_pitch, min_pitch))
    # print("Tilt angles (max, min): ", (max(tilt_angles), min(tilt_angles)))
    print("--------")


if __name__ == "__main__":
    for task in Tasks.task_6:
        file_name = task
        panda_traj = get_rot_panda_traj(file_name)
        # print first step of the euler angles
        # print(quat_to_euler(panda_traj[0, 0, 3:7]))

        # get_tilt_angles(panda_traj)
        # get_max_roll_pitch(panda_traj)

        alex_method(panda_traj)
        # rotation_matrixes(panda_traj)
        # transform_matrices(panda_traj)

# tilt angles
# task 1: 29, 20, 13, 18, 13 // allowed: 39
# task 2: 14, 20, 41, 22, 40 // allowed: 65
# task 3: 32, 22, 78, 80, 13 // allowed: 49
# task 4: 90, 13, 13, 82, 33 // allowed: 72
# task 5: 102, 15, 13, 29, 13 // allowed: 35
# task 6: 24, 100, 24, 32, 86 // allowed: 62