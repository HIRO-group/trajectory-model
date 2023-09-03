from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import numpy as np

from read import read_vectors
from jacobian import get_jacobian


def get_ee_vel(file_name):
    file_path_v = '/home/ava/projects/assets/'+file_name+'/joint_velocities.bin'
    file_path_p = '/home/ava/projects/assets/'+file_name+'/joint_waypoints.bin'
    joint_velocities = read_vectors(file_path_v)
    joint_poisitions = read_vectors(file_path_p)

    ee_velocities = []
    for idx, vel in enumerate(joint_velocities):
        J = get_jacobian(joint_poisitions[idx])
        ee_vel = J.dot(vel)
        ee_velocities.append(ee_vel)

    x_vels, y_vels, z_vels = [], [], []
    for vel in ee_velocities:
        x_vels.append(vel[0])
        y_vels.append(vel[1])
        z_vels.append(vel[2])
    
    return x_vels, y_vels, z_vels


if __name__ == "__main__":
    # file_name = '01-09-2023 13-42-14'
    file_name = '01-09-2023 13-58-43'
    # file_name = "01-09-2023 14-09-56"

    x_vels, y_vels, z_vels = get_ee_vel(file_name)

    plt.subplot(3, 1, 1)
    plt.plot(x_vels)
    plt.ylabel('X')

    plt.subplot(3, 1, 2)
    plt.plot(y_vels)
    plt.ylabel('Y')

    plt.subplot(3, 1, 3)
    plt.plot(z_vels)
    plt.ylabel('Z')

    plt.xlabel('Sample')
    plt.tight_layout()
    # plt.savefig('/home/ava/projects/trajectory-model/plots/velocities/{file_name}.png'.format(file_name=file_name), dpi=300)
    plt.show()
