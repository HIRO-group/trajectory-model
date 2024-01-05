import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from trajectory_model.classifier_predict_func_api import process_panda_to_model_input
from trajectory_model.helper.read import read_panda_vectors
from trajectory_model.helper.helper import quat_to_euler


def get_rot_panda_traj():
    file_name = "21-11-2023 18-03-14"
    panda_file_path =  '/home/ava/projects/assets/cartesian/'+file_name+'/cartesian_positions.bin'
    panda_trajectory = read_panda_vectors(panda_file_path)
    panda_trajectory = process_panda_to_model_input(panda_trajectory)
    panda_trajectory = panda_trajectory[np.newaxis, :, :]
    return panda_trajectory


def plot_euler_angles(panda_traj):
    quatertions = panda_traj[0, :, 3:7]
    euler_angles = []
    for q in quatertions[:100]:
        euler_angles.append(quat_to_euler(q))
    euler_angles = np.array(euler_angles)

    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(euler_angles[:, 0], label='Roll')
    ax[1].plot(euler_angles[:, 1], label='Pitch')
    ax[2].plot(euler_angles[:, 2], label='Yaw')

    # Add labels and legend
    ax[2].set_xlabel('Time Steps')
    ax[0].set_ylabel('Angle (degrees)')
    ax[1].set_ylabel('Angle (degrees)')
    ax[2].set_ylabel('Angle (degrees)')

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    plt.suptitle('Panda Trajectory Euler Angles')
    plt.show()



if __name__ == "__main__":
    panda_traj = get_rot_panda_traj()
    print("panda traj shape: ", panda_traj.shape)
    plot_euler_angles(panda_traj)


