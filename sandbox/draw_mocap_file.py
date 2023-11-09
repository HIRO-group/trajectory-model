import numpy as np
import csv

from trajectory_model.helper.read import read_mocap_file
from trajectory_model.helper.helper import plot_X, plot_multiple_e_ids, plot_multiple_X

if __name__ == "__main__":
    PREFIX = "/home/ava/projects/trajectory-model/data/"
    file = PREFIX + "mocap_new/big/full/spill-free/2023-09-08 20:24:51.csv"
    X = read_mocap_file(file)
    plot_X(X, 0, 0.05)