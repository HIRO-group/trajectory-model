import csv
import numpy as np
from trajectory_model.spill_free.model import TrajectoryClassifier
from trajectory_model.spill_free.constants import DT, EMBED_DIM, NUM_HEADS, FF_DIM,\
      MAX_TRAJ_STEPS, BIG_RADIUS, BIG_HEIGHT, SMALL_RADIUS, \
      SMALL_HEIGHT, BIG_FILL_FULL, BIG_FILL_HALF, \
      SMALL_FILL_FULL, SMALL_FILL_HALF
from trajectory_model.helper.helper import quat_to_euler, euler_to_quat, ctime_str
from trajectory_model.helper.rotate_quaternion import quaternion_to_angle_axis, rotate_quaternion

model = TrajectoryClassifier(max_traj_steps=MAX_TRAJ_STEPS, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, ff_dim=FF_DIM)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.build((None, MAX_TRAJ_STEPS, EMBED_DIM))
model.load_weights("/home/ava/projects/trajectory-model/weights/spill_classifier/best/2023-09-09 14:42:38_epoch_191_best_val_acc_0.93_train_acc_0.92.h5")


import numpy as np
from process_data.process_data_SFC import read_from_files, copy_last_non_zero_value, \
    transform_trajectory, add_equivalent_quaternions, round_down_orientation_and_pos, \
    reverse_y_axis
from trajectory_model.spill_free.constants import EMBED_DIM, BIG_RADIUS, BIG_HEIGHT, SMALL_RADIUS, \
    SMALL_HEIGHT, BIG_FILL_FULL, BIG_FILL_HALF, SMALL_FILL_FULL, SMALL_FILL_HALF
from trajectory_model.helper.helper import plot_X, plot_multiple_e_ids, plot_multiple_X
from trajectory_model.helper.rotate_quaternion import quaternion_to_angle_axis, rotate_quaternion


def read_a_file(file_path, radius, height, fill_level):
    X = np.zeros((1, 10000, EMBED_DIM), dtype=np.float64)
    trajectory_index = 0
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            keys = list(row.values())
            y, x, z = np.float64(keys[1]), np.float64(
                keys[2]), np.float64(keys[3])
            a, b, c, d = np.float64(keys[4]), np.float64(
                keys[5]), np.float64(keys[6]), np.float64(keys[7])
            embedding = np.array(
                [[x, y, z, a, b, c, d, radius, height, fill_level]])
            X[0, trajectory_index, :] = embedding
            # print("embedding: ", embedding)
            trajectory_index += 1
        
    return X

name = '2023-09-08 20:22:09'
file_path = f'/home/ava/projects/trajectory-model/data/mocap_new/big/full/spill-free/{name}.csv'
X = read_a_file(file_path, BIG_RADIUS, BIG_HEIGHT, BIG_FILL_FULL)


dt = 600

X = X[:, 0:10000:dt, :]
X = X[:, 0:MAX_TRAJ_STEPS, :]

X = copy_last_non_zero_value(X)
X = transform_trajectory(X)
X = round_down_orientation_and_pos(X)

x = X[0]

prediction = model.predict(x[np.newaxis, 0])
print("prediction: ", prediction)
    # dt*=2

# prediction = model.predict(X_1)
# print("prediction: ", prediction)


# plot_X(X, 0, 0.1)
