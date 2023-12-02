import numpy as np
from trajectory_model.spill_free.model import TrajectoryClassifier
from trajectory_model.spill_free.constants import EMBED_DIM, NUM_HEADS, FF_DIM,\
      MAX_TRAJ_STEPS, BIG_DIAMETER, BIG_HEIGHT, SMALL_DIAMETER, \
      SMALL_HEIGHT, BIG_FILL_FULL, BIG_FILL_HALF, \
      SMALL_FILL_FULL, SMALL_FILL_HALF, ROBOT_DT
from trajectory_model.helper.helper import quat_to_euler, euler_to_quat, ctime_str
from trajectory_model.helper.rotate_quaternion import quaternion_to_angle_axis, rotate_quaternion

model = TrajectoryClassifier(max_traj_steps=MAX_TRAJ_STEPS, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, ff_dim=FF_DIM)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.build((None, MAX_TRAJ_STEPS, EMBED_DIM))
# model.load_weights("/home/ava/projects/trajectory-model/weights/spill_classifier/best/2023-09-09 14:42:38_epoch_191_best_val_acc_0.93_train_acc_0.92.h5")

# radius, height, fill_level


def convert_to_model_input(trajectory):
    trajectory = np.array([np.array(trajectory[i]) for i in range(0, len(trajectory), ROBOT_DT)])
    trajectory = trajectory[0:MAX_TRAJ_STEPS, :]

    properties = np.array([SMALL_DIAMETER, SMALL_HEIGHT, SMALL_FILL_FULL])
    properties = properties[None,:].repeat(trajectory.shape[0],axis=0)
    trajectory = np.concatenate((trajectory, properties), axis=1)
    trajectory = trajectory.reshape(1, MAX_TRAJ_STEPS, EMBED_DIM)
    return trajectory


def translate(trajectory):
    xyz = trajectory[0, :, 0:3] # shape: (T, 3)
    xyz = xyz - trajectory[0, 0, 0:3] # shape: (T, 3)
    trajectory[0, :, 0:3] = xyz
    trajectory[0, :, 1] = -trajectory[0, :, 1] # because that's how the data was trained unfortunately
    return trajectory


def rotate(trajectory):
    q_start = (trajectory[0, 0, 3], trajectory[0, 0, 4], trajectory[0, 0, 5], trajectory[0, 0, 6])
    q = (-0.0045020487159490585, -0.001336313085630536, -0.21967099606990814, -0.9755628108978271)
    angle, axis = quaternion_to_angle_axis(q, q_start)
    
    for step in range(trajectory.shape[1]):
        a, b, c, d = trajectory[0, step, 3], trajectory[0, step, 4], trajectory[0, step, 5], trajectory[0, step, 6]
        x_rot, y_rot, z_rot, w_rot = rotate_quaternion((a, b, c, d), angle, axis)
        trajectory[0, step, 3], trajectory[0, step, 4], trajectory[0, step, 5], trajectory[0, step, 6] = x_rot, y_rot, z_rot, w_rot

    return trajectory


def transform_trajectory(trajectory):
    trajectory = translate(trajectory)
    trajectory = rotate(trajectory)

    for e_id in range(trajectory.shape[0]):
        for i in range(trajectory.shape[1]):
            trajectory[e_id, i, 0:3] = np.round(trajectory[e_id, i, 0:3], 2)
            trajectory[e_id, i, 3:7] = np.round(trajectory[e_id, i, 3:7], 2)
    return trajectory


# def save_trajectory(trajectory, prediction):
#     name = f'data_{round(prediction, 2)}_{ctime_str()}'
#     path = f'/home/ava/projects/trajectory-model/data/panda_ompl/{name}'
#     trajectory = trajectory.reshape(MAX_TRAJ_STEPS, EMBED_DIM)
#     np.savetxt(path, trajectory, delimiter=',')
#     print(f'successfully saved to file: {name}')


def spilled(trajectory):
    trajectory = convert_to_model_input(trajectory)
    trajectory = transform_trajectory(trajectory)
    prediction = model.predict(trajectory)[0][0]
    print("prediction in spill-free: ", prediction, "spilled: ", prediction >= 0.5)
    return prediction >= 0.5
 