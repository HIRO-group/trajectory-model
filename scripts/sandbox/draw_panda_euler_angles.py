import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from trajectory_model.classifier_predict_func_api import process_panda_to_model_input
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

def plot_euler_angles(panda_traj):
    quatertions = panda_traj[0, :, 3:7]
    euler_angles = []
    for q in quatertions:
        euler = quat_to_euler(q)
        euler_angles.append(euler)

    euler_angles = np.array(euler_angles)
    time_steps = [i/1000 for i in range(euler_angles.shape[0])]
    plt.plot(time_steps, euler_angles[:, 0], label='Roll', linestyle='-', color='purple')
    plt.plot(time_steps, euler_angles[:, 1], label='Pitch', linestyle='-', color='orange')
    plt.plot(time_steps, euler_angles[:, 2], label='Yaw', linestyle='-', color='green')


    plt.xlabel('Time Steps (s)')
    plt.ylabel('Angle (degrees)')
    plt.legend()
    plt.show()


def plot_histogram(all_panda_trajs):
    all_euler_angles = []
    for traj in all_panda_trajs:
        quatertions = traj[0, :, 3:7]
        # euler_angles = []
        for q in quatertions:
            euler_angle = R.from_quat(q).as_euler('xyz', degrees=True)
            all_euler_angles.append(np.array(euler_angle))
        # print("len(euler_angles): ", len(euler_angles))
        # all_euler_angles.append(euler_angles)

    print(len(all_euler_angles))
    all_euler_angles = np.array(all_euler_angles)
    print(all_euler_angles.shape)

    roll = all_euler_angles[:, 0]
    pitch = all_euler_angles[:, 1]
    yaw = all_euler_angles[:, 2]
    # roll = roll.flatten()
    # pitch = pitch.flatten()
    # yaw = yaw.flatten()
    fig, axs = plt.subplots(3)
    axs[0].hist(roll, bins=100, color='darkturquoise')
    axs[0].legend(['Roll'])
    axs[1].hist(pitch, bins=100, color='skyblue')
    axs[1].legend(['Pitch'])
    axs[2].hist(yaw, bins=100, color='steelblue')
    axs[2].legend(['Yaw'])
    plt.show()


def plot_euler_angles_all(all_panda_trajs):
    task_2 = "21-11-2023 14-04-04" # pink
    task_4 = "21-11-2023 15-51-43" # green
    task_5 = "21-11-2023 17-39-10" # blue
    
    fig, axs = plt.subplots(3)
    for traj in all_panda_trajs:
        quatertions = traj[0, :, 3:7]
        euler_angles = []
        for q in quatertions:
            euler = quat_to_euler(q)
            euler_angles.append(euler)

        euler_angles = np.array(euler_angles)
        time_steps = [i/1000 for i in range(euler_angles.shape[0])]

        # decrease line thickness

        axs[0].plot(time_steps, euler_angles[:, 0], linestyle='-', color='black', alpha = 0.3, linewidth=0.5)
        axs[1].plot(time_steps, euler_angles[:, 1], linestyle='-', color='black', alpha = 0.3, linewidth=0.5)
        axs[2].plot(time_steps, euler_angles[:, 2], linestyle='-', color='black', alpha = 0.3, linewidth=0.5)

    # task_2 = "21-11-2023 14-04-04" # pink
    # task_4 = "21-11-2023 15-51-43" # green
    # task_5 = "21-11-2023 17-39-10" # blue    
    # panda_traj = get_panda_traj(task_2)
    # quatertions = panda_traj[0, :, 3:7]
    # euler_angles = []
    # for q in quatertions:
    #     euler = quat_to_euler([q[1], q[2], q[3], q[0]])
    #     euler_angles.append(euler)
    # euler_angles = np.array(euler_angles)
    # time_steps = [i/1000 for i in range(euler_angles.shape[0])]
    # axs[0].plot(time_steps, euler_angles[:, 0], linestyle='-', color='pink', )
    # axs[1].plot(time_steps, euler_angles[:, 1], linestyle='-', color='pink', )
    # axs[2].plot(time_steps, euler_angles[:, 2], linestyle='-', color='pink', )

    # panda_traj = get_panda_traj(task_4)
    # quatertions = panda_traj[0, :, 3:7]
    # euler_angles = []
    # for q in quatertions:
    #     euler = quat_to_euler([q[1], q[2], q[3], q[0]])
    #     euler_angles.append(euler)
    # euler_angles = np.array(euler_angles)
    # time_steps = [i/1000 for i in range(euler_angles.shape[0])]
    # axs[0].plot(time_steps, euler_angles[:, 0], linestyle='-', color='limegreen',)
    # axs[1].plot(time_steps, euler_angles[:, 1], linestyle='-', color='limegreen',)
    # axs[2].plot(time_steps, euler_angles[:, 2], linestyle='-', color='limegreen',)
    
    # panda_traj = get_panda_traj(task_5)
    # quatertions = panda_traj[0, :, 3:7]
    # euler_angles = []
    # for q in quatertions:
    #     euler = quat_to_euler([q[1], q[2], q[3], q[0]])
    #     euler_angles.append(euler)
    # euler_angles = np.array(euler_angles)
    # time_steps = [i/1000 for i in range(euler_angles.shape[0])]
    # axs[0].plot(time_steps, euler_angles[:, 0], linestyle='-', color='steelblue', )
    # axs[1].plot(time_steps, euler_angles[:, 1], linestyle='-', color='steelblue', )
    # axs[2].plot(time_steps, euler_angles[:, 2], linestyle='-', color='steelblue', )



    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Roll (degrees)')
    # axs[0].legend(['Roll'])
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Pitch (degrees)')
    # axs[1].legend(['Pitch'])
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Yaw (degrees)')
    # axs[2].legend(['Yaw'])
    plt.show()


if __name__ == "__main__":
    # file_name = Tasks.task_5[2]
    # file_name = ['']
    # file_names = ["21-11-2023 14-04-04", "21-11-2023 15-47-39", "21-11-2023 17-39-10"]
    # file_name = "21-11-2023 15-51-43"
    # panda_traj = get_panda_traj(file_name)
    # plot_euler_angles(panda_traj)
    all_panda_trajs = []
    for task in Tasks.all:
        for fn in task:
            panda_traj = get_panda_traj(fn)
            print(panda_traj.shape)
            panda_traj = rotate_panda_to_match_orientation(panda_traj)
            print(panda_traj.shape)
            euler_angles = get_euler_angles(panda_traj)
            roll, pitch, yaw = euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]
            if max(roll) - min(roll) > 100:
                continue
            if max(pitch) - min(pitch) > 100:
                continue
            if max(yaw) - min(yaw) > 100:
                continue
            all_panda_trajs.append(panda_traj)
    # plot_histogram(all_panda_trajs)
    plot_euler_angles_all(all_panda_trajs)