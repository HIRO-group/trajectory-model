import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from trajectory_model.predict_api import process_panda_to_model_input
from trajectory_model.helper.read import read_panda_vectors
from trajectory_model.helper.helper import quat_to_euler, get_start_and_end_points

from sandbox.constants import Tasks


def get_rot_panda_traj():
    file_name = Tasks.task_6[1]
    panda_file_path =  '/home/ava/projects/assets/cartesian/'+file_name+'/cartesian_positions.bin'
    panda_trajectory = read_panda_vectors(panda_file_path)
    panda_trajectory = process_panda_to_model_input(panda_trajectory)
    panda_trajectory = panda_trajectory[np.newaxis, ::2, :]
    return panda_trajectory


def plot_panda_traj(panda_traj):
    start_points, end_points = get_start_and_end_points(panda_traj, 0)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(start_points[:, 0], start_points[:, 1], start_points[:, 2],
                end_points[:, 0], end_points[:, 1], end_points[:, 2],
                length = 0.05, normalize = True)
    

    quatertions = panda_traj[0, :, 3:7]
    euler_angles = []
    for q in quatertions:
        euler_angles.append(quat_to_euler(q))
    euler_angles = np.array(euler_angles)

    # plot the row, pitch, and yaw angles above each quiver
    for i in range(len(start_points)):
        ax.text(start_points[i, 0], start_points[i, 1], start_points[i, 2],
                f'({round(euler_angles[i, 0], 2)})')

    # plt.plot(euler_angles[:, 0], label='Roll')
    # plt.plot(euler_angles[:, 1], label='Pitch')
    # plt.plot(euler_angles[:, 2], label='Yaw')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()



if __name__ == "__main__":
    panda_traj = get_rot_panda_traj()
    # panda_traj = panda_traj[::50, :, :]
    # print("panda_traj.shape: ", panda_traj.shape)
    plot_panda_traj(panda_traj)