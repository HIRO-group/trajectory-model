import numpy as np

from trajectory_model.informed_sampler.model import PositionSampler, OrientationSampler
from trajectory_model.informed_sampler.constants import EMBED_DIM_ORIENTATION_X, \
                                        EMBED_DIM_ORIENTATION_Y, MAX_NUM_WAYPOINTS, \
                                        EMBED_DIM_POS_X, EMBED_DIM_POS_Y, MAX_NUM_WAYPOINTS


position_model = PositionSampler(max_num_waypoints=MAX_NUM_WAYPOINTS,
                                  embed_X=EMBED_DIM_POS_X, 
                                  embed_Y=EMBED_DIM_POS_Y)
position_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
position_model.build((None, MAX_NUM_WAYPOINTS, EMBED_DIM_POS_X))
position_model.load_weights("/home/ava/projects/trajectory-model/weights/position_sampler/val_0.89_train_0.92.h5")


orientation_model = OrientationSampler(max_num_waypoints=MAX_NUM_WAYPOINTS,
                               embed_X=EMBED_DIM_ORIENTATION_X,
                               embed_Y=EMBED_DIM_ORIENTATION_Y)
orientation_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
orientation_model.build((None, MAX_NUM_WAYPOINTS, EMBED_DIM_ORIENTATION_X))
orientation_model.load_weights("/home/ava/projects/trajectory-model/weights/orientation_sampler/09:18:26_epoch_596_best_val_acc_0.89_train_acc_0.79.h5")


def get_position_data(trajectory):
    trajectory_pos = []
    for traj in trajectory:
        trajectory_pos.append(traj[0:3])
    return trajectory_pos


def translate(trajectory):
    xyz = trajectory[0, :, 0:3] # shape: (T, 3)
    xyz = xyz - trajectory[0, 0, 0:3] # shape: (T, 3)    
    trajectory[0, :, 0:3] = xyz
    return trajectory


def convert_to_position_model_input(trajectory):
    step_size = int(len(trajectory) / MAX_NUM_WAYPOINTS)
    if step_size == 0: 
        step_size = 1
    
    remainder = len(trajectory) % MAX_NUM_WAYPOINTS

    if len(trajectory) < MAX_NUM_WAYPOINTS:
        remainder = 0
    trajectory = np.array([np.array(trajectory[i]) for i in range(0, len(trajectory) - remainder, step_size)])
    
    if len(trajectory) < MAX_NUM_WAYPOINTS:
        trajectory = np.append(trajectory, np.zeros((MAX_NUM_WAYPOINTS - len(trajectory), 3)), axis=0)

    trajectory = np.append(trajectory, np.ones((trajectory.shape[0], 1)), axis=1)
    trajectory = trajectory.reshape(1, MAX_NUM_WAYPOINTS, EMBED_DIM_POS_X)
    return trajectory


def convert_to_orientation_model_input(trajectory, position_prediction):
    print("trajectory: ", trajectory)
    print("position prediction: ", position_prediction)

    step_size = int(len(trajectory) / MAX_NUM_WAYPOINTS)
    if step_size == 0:
        step_size = 1

    remainder = len(trajectory) % MAX_NUM_WAYPOINTS
    
    if len(trajectory) < MAX_NUM_WAYPOINTS:
        remainder = 0
    trajectory = np.array([np.array(trajectory[i]) for i in range(0, len(trajectory) - remainder, step_size)])
    trajectory = np.vstack((trajectory, np.array([position_prediction[0], position_prediction[1], position_prediction[2], 0, 0, 0, 0])))
    
    if len(trajectory) < MAX_NUM_WAYPOINTS:
        trajectory = np.append(trajectory, np.zeros((MAX_NUM_WAYPOINTS - len(trajectory), 7)), axis=0)

    trajectory = np.append(trajectory, np.ones((trajectory.shape[0], 1)), axis=1)
    trajectory = trajectory.reshape(1, MAX_NUM_WAYPOINTS, EMBED_DIM_ORIENTATION_X)
    return trajectory


def sample_state(trajectory):
    trajectory_pos = get_position_data(trajectory)
    trajectory_pos = convert_to_position_model_input(trajectory_pos)
    trajectory_pos = translate(trajectory_pos)
    position_prediction = position_model.predict(trajectory_pos)[0]
    
    trajectory = convert_to_orientation_model_input(trajectory, position_prediction)
    trajectory = translate(trajectory)
    orientation_prediction = orientation_model.predict(trajectory)[0]

    state = [position_prediction[0], position_prediction[1], position_prediction[2], orientation_prediction[0], orientation_prediction[1], orientation_prediction[2], orientation_prediction[3]]
    print("state cartesian: ", state)
    # should perform ik here

    return [0.5295887589454651,-0.16105137765407562,0.296753466129303,0.004732200410217047,-0.018793363124132156,0.03531074523925781,-0.9991884827613831]