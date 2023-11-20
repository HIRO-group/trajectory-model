import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from process_data.process_data import process_data_SFC, keep_spill_free, compute_delta_X
from trajectory_model.helper.read import read_panda_vectors
from trajectory_model.classifier_predict_func_api import process_panda_to_model_input
from trajectory_model.helper.helper import plot_multiple_X

def get_panda_traj():
    # file_name = '01-09-2023 13-42-14'
    # file_name = '01-09-2023 13-58-43'
    file_name = "01-09-2023 14-09-56"
    panda_file_path =  '/home/ava/projects/assets/cartesian/'+file_name+'/cartesian_positions.bin'
    panda_trajectory = read_panda_vectors(panda_file_path)
    panda_trajectory = process_panda_to_model_input(panda_trajectory)
    panda_trajectory = panda_trajectory[np.newaxis, :, :]
    panda_trajectory = compute_delta_X(panda_trajectory)
    return panda_trajectory

def get_mocap_traj():
    cup_eid = 1
    cup_X, cup_Y = process_data_SFC()
    cup_X, cup_Y = keep_spill_free(cup_X, cup_Y)
    cup_X = cup_X[cup_eid, :, :]
    cup_X = cup_X[np.newaxis, :, :]
    cup_X = compute_delta_X(cup_X)
    return cup_X

if __name__ == "__main__":
    panda_traj = get_panda_traj()
    cup_traj = get_mocap_traj()
    # plot_multiple_X([panda_traj, cup_traj], [0, 0], 0.03, verbose=True)
    
    positions = cup_traj[0, :, :3]
    positions = positions[positions[:, 0] < 10] 
    positions = positions[positions[:, 1] < 10]
    positions = positions[positions[:, 2] < 10]

    positions = positions[positions[:, 0] > -10] 
    positions = positions[positions[:, 1] > -10]
    positions = positions[positions[:, 2] > -10]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='b', marker='o', label='Trajectory')
    
    positions = panda_traj[0, :, :3]
    positions = positions[positions[:, 0] < 10] 
    positions = positions[positions[:, 1] < 10]
    positions = positions[positions[:, 2] < 10]

    positions = positions[positions[:, 0] > -10] 
    positions = positions[positions[:, 1] > -10]
    positions = positions[positions[:, 2] > -10]
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='g', marker='o', label='Trajectory')

    # Customize the plot
    ax.set_title('3D Trajectory Scatter Plot')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.legend()

    # Show the plot
    plt.show()


