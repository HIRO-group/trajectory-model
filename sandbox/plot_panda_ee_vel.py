from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import numpy as np

from trajectory_model.helper.read import read_panda_vectors
from trajectory_model.helper.jacobian import get_jacobian


def get_ee_vel(file_name):
    file_path_v = '/home/ava/projects/assets/'+file_name+'/joint_velocities.bin'
    file_path_p = '/home/ava/projects/assets/'+file_name+'/joint_waypoints.bin'
    joint_velocities = read_panda_vectors(file_path_v)
    joint_poisitions = read_panda_vectors(file_path_p)

    ee_velocities = []
    for idx, vel in enumerate(joint_velocities):
        J = get_jacobian(joint_poisitions[idx])
        ee_vel = J.dot(vel)
        ee_velocities.append(ee_vel)

    x_vels, y_vels, z_vels = [], [], []
    vels = []
    for vel in ee_velocities:
        x_vels.append(vel[0])
        y_vels.append(vel[1])
        z_vels.append(vel[2])
        # sum x_vel, y_vel, z_vel
        vels.append(np.linalg.norm(vel))
    
    return x_vels, y_vels, z_vels, vels


if __name__ == "__main__":
    # file_name = '01-09-2023 13-42-14'
    # file_name = '01-09-2023 13-58-43'
    # # file_name = "01-09-2023 14-09-56"

    small_full_file_names = ["10-09-2023 13-14-07", "10-09-2023 13-10-26", "10-09-2023 10-03-18",
                    "10-09-2023 13-12-16", "10-09-2023 12-36-43"]
    small_half_file_names = ["10-09-2023 13-30-09", "10-09-2023 13-32-29", "10-09-2023 13-39-37"]

    big_half_file_names = ["01-09-2023 13-42-14", "01-09-2023 13-58-43", "01-09-2023 14-09-56"]

    big_full_file_names = ["10-09-2023 10-06-37", "10-09-2023 12-25-04", "10-09-2023 12-29-22"]


    file_names = [small_full_file_names[0], small_half_file_names[0], 
                big_full_file_names[0], big_half_file_names[0]]

    experiment_names = ["Champagne 0.8", "Champagne 0.5", "Wine 0.8", "Wine 0.3"]
    
    x_vels, y_vels, z_vels = [], [], []
    vels = []

    for file_name in file_names:
        rolls, pitches, yaws, vel = get_ee_vel(file_name)
        x_vels.append(rolls)
        y_vels.append(pitches)
        z_vels.append(yaws)
        vels.append(vel)

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # for i in range(4):
    #     axs[0].plot(x_vels[i], label=experiment_names[i])

    # # for i in range(4):
    # #     axs[1].plot(y_vels[i], label=experiment_names[i])

    # # for i in range(4):
    # #     axs[2].plot(z_vels[i], label=experiment_names[i])

    for i in range(4):
        axs[0].plot(vels[i], label=experiment_names[i])

    for ax in axs:
        ax.legend()

    axs[0].set_ylabel("X velocity")
    # axs[1].set_ylabel("Y velocity")
    # axs[2].set_ylabel("Z velocity")
    axs[2].set_xlabel("Time")

    plt.suptitle("ee velocities from 4 experiments")
    plt.savefig(f'plots/velocities/{file_names}.png'.format(file_names=','.join(file_names)), dpi=300)
    plt.show()

    # x_vels, y_vels, z_vels = get_ee_vel(file_name)

    # plt.subplot(3, 1, 1)
    # plt.plot(x_vels)
    # plt.ylabel('X')

    # plt.subplot(3, 1, 2)
    # plt.plot(y_vels)
    # plt.ylabel('Y')

    # plt.subplot(3, 1, 3)
    # plt.plot(z_vels)
    # plt.ylabel('Z')

    # plt.xlabel('Sample')
    # plt.tight_layout()
    # # plt.savefig('/home/ava/projects/trajectory-model/plots/velocities/{file_name}.png'.format(file_name=file_name), dpi=300)
    # plt.show()
