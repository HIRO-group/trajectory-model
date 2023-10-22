import numpy as np
from trajectory_model.spill_free.constants import MAX_TRAJ_STEPS, EMBED_DIM, DT, \
    BIG_RADIUS, BIG_HEIGHT, SMALL_RADIUS, SMALL_HEIGHT, \
    BIG_FILL_FULL, BIG_FILL_HALF, SMALL_FILL_FULL, SMALL_FILL_HALF

from process_data.process_data_SFC import read_from_files, copy_last_non_zero_value, \
    transform_trajectory, add_equivalent_quaternions, round_down_orientation_and_pos, \
    reverse_y_axis

from trajectory_model.helper.helper import plot_multiple_e_ids

FILE_NAMES_NOSPILL_SPILL = \
    [("big/half-full/spill-free/", "big/half-full/spilled/", BIG_RADIUS, BIG_HEIGHT, BIG_FILL_HALF)]

def keep_spill_free(X, Y):
    X_new, Y_new = [], []
    for e_id in range(0, X.shape[0]):
        if Y[e_id] == 0:
            X_new.append(X[e_id])
            Y_new.append(Y[e_id])
    X_new = np.array(X_new)
    Y_new = np.array(Y_new)
    return X_new, Y_new


X, Y = read_from_files(file_list = FILE_NAMES_NOSPILL_SPILL)
X, Y = keep_spill_free(X, Y)
X = copy_last_non_zero_value(X)
X = transform_trajectory(X)
# X, Y = add_equivalent_quaternions(X, Y)
X = round_down_orientation_and_pos(X)
X = reverse_y_axis(X)

X = X[:, 40:X.shape[1]-70:20, :]

x_min = np.min(X[:, :, 0])
y_min = np.min(X[:, :, 1])
z_min = np.min(X[:, :, 2])
x_max = np.max(X[:, :, 0])
y_max = np.max(X[:, :, 1])
z_max = np.max(X[:, :, 2])

# # generate random arrows in this range
# X = np.zeros((X.shape[0], X.shape[1], EMBED_DIM), dtype=np.float64)
# for e_id in range(X.shape[0]):
    



plot_multiple_e_ids(X, [i for i in range(0, 20)], 0.1)


