import numpy as np
import matplotlib.pyplot as plt


def get_initials(data_num):
    if data_num == 0:
        x, y, z = 0.2, -1, 0.1
        xdot, ydot, zdot = 0, 0, 0
        xddot, yddot, zddot = 0, 0.4, 0.3
        y_train = 0 # slosh free
    else:
        x, y, z = 0.2, -1, 0.1
        xdot, ydot, zdot = 0, 0, 0
        xddot, yddot, zddot = 1, 3, 10
        y_train = 1 # slosh
    return y_train, x, y, z, xdot, ydot, zdot, xddot, yddot, zddot


def generate_fake_data(data_num, max_traj_steps, embed_dim, dt):
    # Rough estimate of Panda's reachable area: x: (0, 1.57), y: (-1, 1), z: (0, 1)
    x_train = np.zeros((data_num, max_traj_steps, embed_dim), dtype=np.float64)
    y_train = np.zeros((data_num, 1))
    
    phi, theta, psi = 0, 0, 0
    phidot, thetadot, psidot = 0, 0, 0
    cup_type = 1

    for d in range(data_num):
        y_train[d], x, y, z, xdot, ydot, zdot, xddot, yddot, zddot = get_initials(d)
        zddot_slow_down = 3.7*zddot/(max_traj_steps//2)
        
        for i in range(max_traj_steps):
            x_train[d, i, :] = np.array([x, xdot, xddot, y, ydot, yddot, z, zdot, zddot, phi, phidot, theta, thetadot, psi, psidot, cup_type])

            x += xdot * dt
            y += ydot * dt
            z += zdot * dt
            xdot += xddot * dt
            ydot += yddot * dt
            zdot += zddot * dt
            
            if i >= max_traj_steps//4 and i <= 3 * max_traj_steps//4:
                zddot -= zddot_slow_down
    return x_train, y_train


def visualize_data(x_train):
    d = 0
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([0, 1.57])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1])
    ax.scatter(x_train[d, :, 0], x_train[d, :, 3], x_train[d, :, 6])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # plt.show()
    # plt.savefig('data_safe_trajectory.png')

def get_data(data_num, max_traj_steps, embed_dim, dt, debug=False):
    if debug:
        x_train, y_train = generate_fake_data(data_num, max_traj_steps, embed_dim, dt)
        x_val, y_val = x_train, y_train
        return x_train, y_train, x_val, y_val
    else:
        #have to make sure that the data is between a certain range so positional encoding works properly
        raise NotImplementedError