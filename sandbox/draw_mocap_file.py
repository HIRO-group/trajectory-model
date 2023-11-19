import numpy as np
import csv

from trajectory_model.helper.read import read_mocap_file
from trajectory_model.helper.helper import plot_X, plot_multiple_e_ids, plot_multiple_X, quat_to_euler

if __name__ == "__main__":
    # static
    # file_st_up = "/home/ava/projects/trajectory-model/data/mocap_new_cups/tumbler/30/spilled/2023-11-14 15:40:04.csv"
    # file_st_dn = "/home/ava/projects/trajectory-model/data/mocap_new_cups/tumbler/30/spilled/2023-11-14 15:40:48.csv"
    # # go ...
    # file_up = "/home/ava/projects/trajectory-model/data/mocap_new_cups/tumbler/30/spilled/2023-11-14 15:48:13.csv"
    # file_down = "/home/ava/projects/trajectory-model/data/mocap_new_cups/tumbler/30/spilled/2023-11-14 15:51:11.csv"
    # file_left = "/home/ava/projects/trajectory-model/data/mocap_new_cups/tumbler/30/spilled/2023-11-14 15:54:38.csv"
    # file_right = "/home/ava/projects/trajectory-model/data/mocap_new_cups/tumbler/30/spilled/2023-11-14 15:54:57.csv"

    # X_up = read_mocap_file(file_up) 
    # X_down = read_mocap_file(file_down)
    # X_left = read_mocap_file(file_left)
    # X_right = read_mocap_file(file_right)
    # X_st_up = read_mocap_file(file_st_up)
    # X_st_down = read_mocap_file(file_st_dn)
    # plot_multiple_X([X_up, X_down, X_left, X_right, X_st_up, X_st_down], [0, 0, 0, 0, 0, 0], 0.1)
    
    # X_st_up = X_st_up[:, 0, :]
    # X_st_down = X_st_down[:, 0, :]
    # X_st_up = X_st_up[:, np.newaxis, :]
    # X_st_down = X_st_down[:, np.newaxis, :]
    # print("static up euler angles: ", quat_to_euler(X_st_up[0, 0, 3:7]))
    # print("static down euler angles: ", quat_to_euler(X_st_down[0, 0, 3:7]))
    # print("diff: ", quat_to_euler(X_st_up[0, 0, 3:7]) - quat_to_euler(X_st_down[0, 0, 3:7]))

    # plot_multiple_X([X_st_up, X_st_down], [0, 0], 0.1)
    # plot_multiple_X([X_st_up], [0], 0.00001)
    # plot_multiple_X([X_st_down], [0], 0.00001)

    # file_normal_st_up = "/home/ava/projects/trajectory-model/data/mocap_new/up.csv"
    # file = read_mocap_file(file_normal_st_up)
    # plot_multiple_X([file], [0], 0.00001)

    # file = "/home/ava/projects/trajectory-model/data/mocap_new_cups/test_down.csv"
    # file = "/home/ava/projects/trajectory-model/data/mocap_new_cups/shorttumbler/70/spill-free/2023-11-16 13:04:29.csv"
    # file = "/home/ava/projects/trajectory-model/data/mocap_new_cups/shorttumbler/70/spill-free/2023-11-16 13:01:44.csv"
    # file = "/home/ava/projects/trajectory-model/data/mocap_new_cups/shorttumbler/30/spilled/2023-11-16 12:52:27.csv"
    # file = "/home/ava/projects/trajectory-model/data/mocap_new_cups/shorttumbler/30/spilled/2023-11-16 12:52:51.csv"
    file = "/home/ava/projects/trajectory-model/data/mocap_new_cups/shorttumbler/30/spilled/2023-11-16 12:54:50.csv"
    X = read_mocap_file(file)
    plot_multiple_X([X], [0], 0.01)