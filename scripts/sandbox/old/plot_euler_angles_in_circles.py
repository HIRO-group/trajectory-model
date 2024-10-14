import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from trajectory_model.predict_api import process_panda_to_model_input
from trajectory_model.helper.read import read_panda_vectors
from trajectory_model.helper.helper import quat_to_euler
from trajectory_model.helper.rotate_quaternion import rotate_panda_to_match_orientation

from sandbox.constants import Tasks

def get_panda_traj(file_name):
    panda_file_path = '/home/ava/projects/assets/cartesian/'+file_name+'/cartesian_positions.bin'
    trajectory = read_panda_vectors(panda_file_path)
    trajectory = np.array([np.array(trajectory[i]) for i in range(0, len(trajectory), 1)])
    trajectory = trajectory[np.newaxis, :, :]
    return trajectory

def get_euler_angles(panda_traj):
    quatertions = panda_traj[0, :, 3:7]
    euler_angles = []
    for q in quatertions:
        euler = quat_to_euler(q)
        euler_angles.append(euler)
    return np.array(euler_angles)


def plot_euler_angles_for_a_task(roll_range, pitch_range, yaw_range):
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})

    # fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # Convert angle ranges to radians
    roll_min_rad, roll_max_rad = np.radians(roll_range)
    pitch_min_rad, pitch_max_rad = np.radians(pitch_range)
    yaw_min_rad, yaw_max_rad = np.radians(yaw_range)

    # Plot angle ranges
    ax.fill_between([roll_min_rad, roll_max_rad], [1, 1], color='red', alpha=0.3, label='Roll')
    ax.fill_between([pitch_min_rad, pitch_max_rad], [1, 1], color='blue', alpha=0.3, label='Pitch')
    ax.fill_between([yaw_min_rad, yaw_max_rad], [1, 1], color='green', alpha=0.3, label='Yaw')
    ax.grid(False)
    ax.set_yticklabels([])
    # ax.set_xticklabels(['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$',
    #                     r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$'])
    ax.legend()
    # plt.title('Angle Ranges on Circle')
    # plt.show()
    plt.savefig('angle_ranges_on_circle_degree.png')


def get_euler_angles_roll_range(task):
    min_roll, max_roll = 360, -360
    min_pitch, max_pitch = 360, -360
    min_yaw, max_yaw = 360, -360

    for trial in task:
        panda_traj = get_panda_traj(trial)
        euler_angles = get_euler_angles(panda_traj)

        roll, pitch, yaw = euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]
        if max(roll) - min(roll) > 100:
            continue
        if max(pitch) - min(pitch) > 100:
            continue
        if max(yaw) - min(yaw) > 100:
            continue

        if min(roll) < min_roll:
            min_roll = min(roll)
        if max(roll) > max_roll:
            max_roll = max(roll)
        if min(pitch) < min_pitch:
            min_pitch = min(pitch)
        if max(pitch) > max_pitch:
            max_pitch = max(pitch)
        if min(yaw) < min_yaw:
            min_yaw = min(yaw)
        if max(yaw) > max_yaw:
            max_yaw = max(yaw)

        roll_range = (min_roll, max_roll)
        pitch_range = (min_pitch, max_pitch)
        yaw_range = (min_yaw, max_yaw)

    print("roll range: ", roll_range)
    print("pitch range: ", pitch_range)
    print("yaw range: ", yaw_range)
    return roll_range, pitch_range, yaw_range


def plot_euler_angles_on_range_chart(roll_range, pitch_range, yaw_range):
    fig, ax = plt.subplots(figsize=(1, 0.2))
    
    # Plot roll range on y-axis
    ax.plot(roll_range, [0.1, 0.1], 'violet', label='roll', linewidth=5)
    
    # Plot pitch range on y-axis   
    ax.plot( pitch_range, [0.105, 0.105], 'pink', label='pitch', linewidth=5)

    # Plot yaw range on y-axis
    ax.plot(yaw_range, [0.11, 0.11], 'orange', label='yaw', linewidth=5)

    # write roll, pitch, yaw on the middle of the range


    ax.text(1/2*(roll_range[0] + roll_range[1]), 0.1, 'roll', fontsize=12, ha='center')
    ax.text(1/2*(pitch_range[0] + pitch_range[1]), 0.105, 'pitch', fontsize=12, ha='center')
    ax.text(1/2*(yaw_range[0] + yaw_range[1]), 0.11, 'yaw', fontsize=12, ha='center')
    # ax.text(pitch_range[0], 0.11, 'pitch', fontsize=12, ha='right')
    # ax.text(yaw_range[0], 0.12, 'yaw', fontsize=12, ha='right')

    ax.set_ylim(0.09, 0.13) # Tighten x-axis limits
    # ax.set_ylim(-180, 180) # Tighten y-axis limits
    ax.set_xticks([i for i in range(-180, 90, 30)])
    # ax.set_xticks()
    # ax.set_xticks([0, 1, 2]) 
    # ax.set_xticks([0, 1, 2])
    ax.set_yticklabels([])
    
    # ax.legend()
    
    plt.show()


if __name__ == "__main__":
    task = []
    task =  Tasks.task_1 + Tasks.task_2 + Tasks.task_3 + Tasks.task_4 + Tasks.task_5 + Tasks.task_6
    roll_range, pitch_range, yaw_range = get_euler_angles_roll_range(task)
    # plot_euler_angles_for_a_task(roll_range, pitch_range, yaw_range)
    plot_euler_angles_on_range_chart(roll_range, pitch_range, yaw_range)