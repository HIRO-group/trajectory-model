import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from trajectory_model.predict_api import process_panda_to_model_input
from trajectory_model.helper.read import read_panda_vectors
from trajectory_model.helper.helper import quat_to_euler
from trajectory_model.helper.rotate_quaternion import rotate_panda_to_match_orientation

from sandbox.constants import Tasks

def get_rot_panda_traj():
    # file_name = "21-11-2023 18-03-14"
    file_name = Tasks.task_6[1]
    panda_file_path = '/home/ava/projects/assets/cartesian/'+file_name+'/cartesian_positions.bin'
    trajectory = read_panda_vectors(panda_file_path)
    trajectory = np.array([np.array(trajectory[i]) for i in range(0, len(trajectory), 1)])
    # trajectory = rotate_panda_to_match_orientation(trajectory)

    trajectory = trajectory[np.newaxis, :, :]
    return trajectory


def plot_euler_angles(panda_traj):
    quatertions = panda_traj[0, :, 3:7] # 2.5 to 4.2 = 1.7 2500:4100
    euler_angles = []
    for q in quatertions:
        euler = quat_to_euler(q)
        euler_angles.append(euler)


    euler_angles = np.array(euler_angles)
    time_steps = [i/1000 for i in range(euler_angles.shape[0])]

    # draw euler angles with similar color but different line style

    plt.plot(time_steps, euler_angles[:, 0], label='Roll', linestyle='-', color='g')
    plt.plot(time_steps, euler_angles[:, 1], label='Pitch', linestyle='--', color='g')
    plt.plot(time_steps, euler_angles[:, 2], label='Yaw', linestyle=':', color='g')


    # plt.plot(time_steps, euler_angles[:, 0], label='Roll')
    # plt.plot(time_steps, euler_angles[:, 1], label='Pitch')
    # plt.plot(time_steps, euler_angles[:, 2], label='Yaw')

    

    plt.xlabel('Time Steps (s)')
    plt.ylabel('Angle (degrees)')
    # plt.yticks(range(-20, 85, 5))
    plt.legend()
    # plt.suptitle('Panda Trajectory Euler Angles')
    plt.show()



if __name__ == "__main__":
    panda_traj = get_rot_panda_traj()
    plot_euler_angles(panda_traj)


