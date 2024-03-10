import numpy as np
from trajectory_model.helper.read import read_panda_vectors
from sandbox.constants import Tasks
from scipy.spatial.transform import Rotation
from math import sin, cos, pi
import roboticstoolbox as rtb

import matplotlib.pyplot as plt

from trajectory_model.helper.rotate_quaternion_new_fk import rotate_ee_matrix


def get_panda_joint_angles(file_name):
    panda_file_path =  '/home/ava/projects/assets/'+file_name+'/joint_waypoints.bin'
    trajectory = read_panda_vectors(panda_file_path)
    trajectory = np.array([np.array(trajectory[i]) for i in range(0, len(trajectory), 1)])
    return trajectory


def get_panda():
    E1 = rtb.ET.tz(0.333)
    E2 = rtb.ET.Rz()
    E3 = rtb.ET.Ry()
    E4 = rtb.ET.tz(0.316)
    E5 = rtb.ET.Rz()
    E6 = rtb.ET.tx(0.0825)
    E7 = rtb.ET.Ry(flip=True)
    E8 = rtb.ET.tx(-0.0825)
    E9 = rtb.ET.tz(0.384)
    E10 = rtb.ET.Rz()
    E11 = rtb.ET.Ry(flip=True)
    E12 = rtb.ET.tx(0.088)
    E13 = rtb.ET.Rx(np.pi)
    E14 = rtb.ET.tz(0.107)
    E15 = rtb.ET.Rz()
    panda = E1 * E2 * E3 * E4 * E5 * E6 * E7 * \
        E8 * E9 * E10 * E11 * E12 * E13 * E14 * E15
    return panda


def extract_pose(end_effector_pose, degrees=True):
    position = end_effector_pose[:3, 3]
    rotation_matrix = end_effector_pose[:3, :3]
    roll, pitch, yaw = Rotation.from_matrix(rotation_matrix).as_euler('xyz', degrees=degrees)
    return position, roll, pitch, yaw


def perform_fk(q_array, degrees=True):
    panda = get_panda()
    fk = panda.eval(q_array)
    return fk


def plot_euler_angle_for_a_traj(euler_angles, axs, color, alpha, linewidth, start=0, end=-1):
    euler_angles = np.array(euler_angles)
    time_steps = [i/1000 for i in range(start, start+len(euler_angles))]
    axs[0].plot(time_steps, euler_angles[:, 0], linestyle='-', color=color, alpha=alpha, linewidth=linewidth)
    axs[1].plot(time_steps, euler_angles[:, 1], linestyle='-', color=color, alpha=alpha, linewidth=linewidth)
    axs[2].plot(time_steps, euler_angles[:, 2], linestyle='-', color=color, alpha=alpha, linewidth=linewidth)


if __name__ == "__main__":
    task_2 = "21-11-2023 14-04-04" # pink
    task_4 = "21-11-2023 15-51-43" # green
    task_5 = "21-11-2023 17-39-10" # blue 
    
    task = Tasks.task_4[0]


    panda_traj_joint_angles = get_panda_joint_angles(task)

    cup_euler_angles = []
    for joint_angle in panda_traj_joint_angles:
        ee_matrix = perform_fk(joint_angle)
        ee_rot_matrix = ee_matrix[:3, :3]
        ee_euler = Rotation.from_matrix(ee_rot_matrix).as_euler('xyz', degrees=True)

        cup_rot_matrix = rotate_ee_matrix(ee_rot_matrix)
        cup_euler = Rotation.from_matrix(cup_rot_matrix).as_euler('xyz', degrees=True)
        cup_euler_angles.append(cup_euler)

    fig, axs = plt.subplots(3)
    plot_euler_angle_for_a_traj(cup_euler_angles, axs, 'black', 0.7, 0.7)
    plt.show()
    

    # position, roll, pitch, yaw = extract_pose(fk)
    # print("End-effector position (m):", position)
    # print("Roll angle (deg):", roll)
    # print("Pitch angle (deg):", pitch)
    # print("Yaw angle (deg):", yaw)


# joint_angle = panda_traj_joint_angles[0]
# trans_matrix = perform_fk(joint_angle)
# rotation_matrix = trans_matrix[:3, :3]

# panda_trajectory = np

# quaternion = Rotation.from_matrix(rotation_matrix).as_quat() # x y z w
# euler = Rotation.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)

# print(quaternion)
# print(euler)



# print(fk)
# joint_angle = [-0.41289399, 0.841556, -1.21653001, -1.45468999, 0.76619699, 1.79543999, -0.89302799]
# joint_angle = [0, -pi/4, 0, -3 * pi/4, 0, pi/2, pi/4]
# print("End-effector position (m):", position)
# print("Roll angle (deg):", roll)
# print("Pitch angle (deg):", pitch)
# print("Yaw angle (deg):", yaw)
