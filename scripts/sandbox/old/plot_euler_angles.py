import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from sandbox.constants import Tasks
from trajectory_model.helper.rotate_ee_matrix import rotate_panda_traj
from trajectory_model.helper.read import read_panda_vectors


def get_panda_cartesian(file_name):
    panda_file_path =  '/home/ava/projects/assets/cartesian_fixed/'+file_name+'/cartesian.bin'
    trajectory = read_panda_vectors(panda_file_path)
    trajectory = np.array([np.array(trajectory[i]) for i in range(0, len(trajectory), 1)])
    return trajectory


def get_a_specific_euler_angle(file_name):
    cartesian_traj = get_panda_cartesian(file_name)
    rot_panda_traj = rotate_panda_traj(cartesian_traj)
    euler_angles = []
    for cartesian in rot_panda_traj:
        ee_matrix = R.from_quat(cartesian[3:]).as_matrix()
        roll, pitch, yaw = R.from_matrix(ee_matrix).as_euler('xyz', degrees=True)
        euler_angles.append([roll, pitch, yaw])
    return euler_angles


def good_data(file_name):
    euler_angles = get_a_specific_euler_angle(file_name)
    euler_angles = np.array(euler_angles)
    roll, pitch, yaw = euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]
    if max(roll) - min(roll) > 100:
        return False
    if max(pitch) - min(pitch) > 100:
        return False
    if max(yaw) - min(yaw) > 100:
        return False
    return True


def get_all_euler_angles():
    all_euler_angles = []
    for tasks in Tasks.all:
        for task in tasks:
            if not good_data(task):
                continue

            cartesian_traj = get_panda_cartesian(task)
            rot_panda_traj = rotate_panda_traj(cartesian_traj)

            euler_angles = []
            for cartesian in rot_panda_traj:
                ee_matrix = R.from_quat(cartesian[3:]).as_matrix()
                roll, pitch, yaw = R.from_matrix(ee_matrix).as_euler('xyz', degrees=True)
                euler_angles.append([roll, pitch, yaw])
            
            all_euler_angles.append(euler_angles)
    all_euler_angles = [np.array(euler_angles) for euler_angles in all_euler_angles]
    return all_euler_angles


def plot_euler_angle_for_a_traj(euler_angles, axs, color, alpha, linewidth, start=0, end=-1):
    euler_angles = np.array(euler_angles)
    euler_angles = euler_angles[start:end]
    time_steps = [i/1000 for i in range(start, start+len(euler_angles))]

    axs[0].plot(time_steps, euler_angles[:, 0], linestyle='-', color=color, alpha=alpha, linewidth=linewidth)
    axs[1].plot(time_steps, euler_angles[:, 1], linestyle='-', color=color, alpha=alpha, linewidth=linewidth)
    axs[2].plot(time_steps, euler_angles[:, 2], linestyle='-', color=color, alpha=alpha, linewidth=linewidth)


def plot_euler_angles_for_all_traj(all_euler_angles):
    fig, axs = plt.subplots(3)
    for euler_angles in all_euler_angles:
        if len(euler_angles) > 6100:
            continue
        plot_euler_angle_for_a_traj(euler_angles, axs, 'black', 0.2, 0.3)
    
    task_2 = "21-11-2023 14-04-04" # pink
    # task_4 = "21-11-2023 15-51-43" # green
    task_5 = "21-11-2023 17-39-10" # blue 


    euler_angle_2 = get_a_specific_euler_angle(task_2)
    # plot_euler_angle_for_a_traj(euler_angle_2, axs, 'lightcoral', alpha=0.5, linewidth=3, start=3651, end=5284)
    plot_euler_angle_for_a_traj(euler_angle_2, axs, 'lightcoral', alpha=1, linewidth=1)

    # euler_angle_4 = get_a_specific_euler_angle(task_4)
    # plot_euler_angle_for_a_traj(euler_angle_4, axs, 'limegreen', alpha=0.5, linewidth=2, start=2606, end=3652)
    # plot_euler_angle_for_a_traj(euler_angle_4, axs, 'limegreen', alpha=1, linewidth=0.5)

    euler_angle_5 = get_a_specific_euler_angle(task_5)
    # plot_euler_angle_for_a_traj(euler_angle_5, axs, 'steelblue', alpha=0.5, linewidth=3, start=0, end=2000)
    plot_euler_angle_for_a_traj(euler_angle_5, axs, 'steelblue', alpha=1, linewidth=1)

    plt.subplots_adjust(hspace=0.005) 
    axs[0].set_ylabel('Roll (deg)')
    axs[0].xaxis.set_visible(False)
    axs[1].xaxis.set_visible(False)
    axs[1].set_ylabel('Pitch (deg)')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Yaw (deg)')

    euler_angle_5 = np.array(euler_angle_5)
    euler_angle_2 = np.array(euler_angle_2)

    axs[0].set_yticks(
        range(int(round(min(axs[0].get_ylim()), 0)), int(round(max(axs[0].get_ylim()), 0)), 15)
    )
    axs[1].set_yticks(
        range(int(round(min(axs[1].get_ylim()), 0)), int(round(max(axs[1].get_ylim()), 0)), 15)
    )

    axs[2].set_yticks(
        range(int(round(min(axs[2].get_ylim()), 0)), int(round(max(axs[2].get_ylim()), 0)), 5)
    )

    # draw two blue dashed lines at indexes 2606, end=3652 and red ones for 3651, end=5284
    # axs[0].axvline(x=3.651, color='red', linestyle='--', linewidth=0.5)
    # axs[0].axvline(x=5.284, color='red', linestyle='--', linewidth=0.5)
    # axs[0].axvline(x=2.606, color='blue', linestyle='--', linewidth=0.5)
    # axs[0].axvline(x=3.652, color='blue', linestyle='--', linewidth=0.5)


    # axs[0].set_yticks([
    #     0,
    #     round(min(axs[0].get_ylim()), 0),
    #     round(max(axs[0].get_ylim()), 0),
    #     round(max(euler_angle_5[0:2000, 0]), 0),
    #     round(min(euler_angle_2[3651:5284, 0]), 0),
    #     ])
    
    # axs[1].set_yticks([
    #     round(min(axs[1].get_ylim()), 0),
    #     0,
    #     round(max(axs[1].get_ylim()), 0),
    #     # round(min(euler_angle_5[:, 1]), 0),
    #     round(max(euler_angle_2[3651:5284, 1]), 0),
    #     ])

    # axs[2].set_yticks([
    #     round(min(axs[2].get_ylim()), 0),
    #     0,
    #     round(max(axs[2].get_ylim()), 0),
    #     round(max(euler_angle_5[0:2000, 2]), 0),
    #     # round(min(euler_angle_2[:, 2]), 0),
    #     ])

    for ax in axs:
        ax.set_ylim(min(ax.get_ylim())-5, max(ax.get_ylim())+5)
        ax.set_xlim(-0.2, 6.2)


    plt.show()


if __name__ == "__main__":
    all_euler_angles = get_all_euler_angles()
    plot_euler_angles_for_all_traj(all_euler_angles)
