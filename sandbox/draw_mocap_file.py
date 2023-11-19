import numpy as np
import csv

from trajectory_model.helper.read import read_mocap_file
from trajectory_model.helper.helper import plot_X, plot_multiple_e_ids, plot_multiple_X, quat_to_euler

if __name__ == "__main__":
    file = "/home/ava/projects/trajectory-model/data/mocap_new_cups/shorttumbler/30/spilled/2023-11-16 12:54:50.csv"
    X = read_mocap_file(file)
    plot_multiple_X([X], [0], 0.01)