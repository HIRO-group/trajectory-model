import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from trajectory_model.predict_api import process_panda_to_model_input
from trajectory_model.helper.read import read_panda_vectors
from trajectory_model.helper.helper import quat_to_euler
# from trajectory_model.helper.rotate_quaternion import rotate_panda_to_match_orientation
from trajectory_model.helper.new_rotate_quaternion import new_rotate_panda_to_match_orientation
from trajectory_model.predict_api import process_panda_to_model_input
from trajectory_model.helper.read import read_panda_vectors
from trajectory_model.helper.helper import quat_to_euler, get_start_and_end_points

from sandbox.constants import Tasks

def get_euler_angles(panda_traj):
    quatertions = panda_traj[:, 3:7]
    euler_angles = []
    for q in quatertions:
        euler = quat_to_euler(q)
        euler_angles.append(euler)
    return np.array(euler_angles)


def get_panda_traj(file_name):
    panda_file_path =  '/home/ava/projects/assets/cartesian/'+file_name+'/cartesian_positions.bin'
    trajectory = read_panda_vectors(panda_file_path)
    trajectory = np.array([np.array(trajectory[i]) for i in range(0, len(trajectory), 1)])
    return trajectory


def get_rot_panda_traj(file_name):
    panda_traj = get_panda_traj(file_name)
    rot_panda_traj = new_rotate_panda_to_match_orientation(panda_traj)
    return rot_panda_traj


def good_data(panda_traj):
    euler_angles = get_euler_angles(panda_traj)
    roll, pitch, yaw = euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]
    if max(roll) - min(roll) > 100:
        return False
    if max(pitch) - min(pitch) > 100:
        return False
    if max(yaw) - min(yaw) > 100:
        return False
    return True

def plot_euler_angle_for_a_traj(traj, axs, color, alpha, linewidth, start=0, end=-1):
    quatertions = traj[start:end, 3:7]
    euler_angles = []
    for q in quatertions:
        euler = quat_to_euler(q)
        euler_angles.append(euler)

    euler_angles = np.array(euler_angles)

    time_steps = [i/1000 for i in range(start, start+len(euler_angles))]

    axs[0].plot(time_steps, euler_angles[:, 0], linestyle='-', color=color, alpha=alpha, linewidth=linewidth)
    axs[1].plot(time_steps, euler_angles[:, 1], linestyle='-', color=color, alpha=alpha, linewidth=linewidth)
    axs[2].plot(time_steps, euler_angles[:, 2], linestyle='-', color=color, alpha=alpha, linewidth=linewidth)


def plot_euler_angles_rot(rot_panda_trajs):
    fig, axs = plt.subplots(3)
    for traj in rot_panda_trajs:
        plot_euler_angle_for_a_traj(traj, axs, 'black', 0.3, 0.3)

    task_2 = "21-11-2023 14-04-04" # pink
    # task_4 = "21-11-2023 15-51-43" # green
    task_5 = "21-11-2023 17-39-10" # blue 
    task_green_ns = "21-11-2023 15-07-13"

    rot_panda_traj = get_rot_panda_traj(task_2)
    plot_euler_angle_for_a_traj(rot_panda_traj, axs, 'lightcoral', alpha=0.5, linewidth=2, start=3651, end=5284)
    plot_euler_angle_for_a_traj(rot_panda_traj, axs, 'lightcoral', alpha=1, linewidth=0.5)


    rot_panda_traj = get_rot_panda_traj(task_green_ns)
    plot_euler_angle_for_a_traj(rot_panda_traj, axs, 'limegreen', alpha=0.5, linewidth=2, start=0, end=2183)
    plot_euler_angle_for_a_traj(rot_panda_traj, axs, 'limegreen', alpha=1, linewidth=0.5)


    # rot_panda_traj = get_rot_panda_traj(task_5)
    # plot_euler_angle_for_a_traj(rot_panda_traj, axs, 'steelblue', alpha=0.5, linewidth=2, start=0, end=2000)
    # plot_euler_angle_for_a_traj(rot_panda_traj, axs, 'steelblue', alpha=1, linewidth=0.5)

    # print euler angle at step 2000
    print(quat_to_euler(rot_panda_traj[0, 3:7]))
    print(quat_to_euler(rot_panda_traj[2000, 3:7]))

    
    plt.subplots_adjust(hspace=0.05) 
    # axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Roll (degrees)')
    axs[0].xaxis.set_visible(False)
    axs[1].xaxis.set_visible(False)
    # axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Pitch (degrees)')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Yaw (degrees)')

    
    axs[0].set_yticks([round(min(axs[0].get_ylim()), 0),0, round(max(axs[0].get_ylim()), 0)])
    axs[1].set_yticks([round(min(axs[1].get_ylim()), 0), 0, round(max(axs[1].get_ylim()), 0)])
    axs[2].set_yticks([round(min(axs[2].get_ylim()), 0),0, round(max(axs[2].get_ylim()), 0)])

    for ax in axs:
        ax.set_ylim(min(ax.get_ylim())-5, max(ax.get_ylim())+5)
        ax.set_xlim(-0.2, 8)

    plt.show()

if __name__ == "__main__":
    all_rot_panda_trajs = []
    for task in Tasks.all:
        for fn in task:
            panda_traj = get_panda_traj(fn)
            if not good_data(panda_traj):
                continue
            rot_panda_traj = new_rotate_panda_to_match_orientation(panda_traj)
            all_rot_panda_trajs.append(panda_traj)
    plot_euler_angles_rot(all_rot_panda_trajs)