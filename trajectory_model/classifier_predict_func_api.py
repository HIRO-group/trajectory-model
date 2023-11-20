import numpy as np
from trajectory_model.spill_free.model_func_api import get_SFC_model
from trajectory_model.spill_free.constants import \
    MAX_TRAJ_STEPS, EMBED_LOC, BLANK_VAL, ROBOT_DT, \
    BIG_RADIUS_B, BIG_HEIGHT, BIG_RADIUS_U, BIG_FILL_80, BIG_FILL_30, \
    SMALL_RADIUS_B, SMALL_HEIGHT, SMALL_RADIUS_U, SMALL_FILL_80, SMALL_FILL_50, \
    SHORT_TUMBLER_RADIUS_B, SHORT_TUMBLER_HEIGHT, SHORT_TUMBLER_RADIUS_U, SHORT_TUMBLER_FILL_30, SHORT_TUMBLER_FILL_70, \
    TALL_TUMBLER_RADIUS_B, TALL_TUMBLER_HEIGHT, TALL_TUMBLER_RADIUS_U, TALL_TUMBLER_FILL_50, TALL_TUMBLER_FILL_80, \
    TUMBLER_RADIUS_B, TUMBLER_HEIGHT, TUMBLER_RADIUS_U, TUMBLER_FILL_30, TUMBLER_FILL_70, \
    WINE_RADIUS_B, WINE_HEIGHT, WINE_RADIUS_U, WINE_FILL_30, WINE_FILL_70
from trajectory_model.helper.rotate_quaternion import rotate_panda_to_match_orientation

model = get_SFC_model()
# address = '/home/ava/projects/trajectory-model/weights/spill_classifier_func_api/best/2023-11-13 17:39:44_epoch_298_train_acc_0.89.h5'
# model.load_weights(address)


def translate(trajectory):
    xyz = trajectory[:, 0:3]
    xyz = xyz - trajectory[0, 0:3]
    trajectory[:, 0:3] = xyz
    return trajectory


def round_down(trajectory):
    for i in range(trajectory.shape[0]):
        trajectory[i, 0:3] = np.round(trajectory[i, 0:3], 2)
        trajectory[i, 3:7] = np.round(trajectory[i, 3:7], 2)
    return trajectory


def delta_xyz(trajectory):
    delta_X = np.copy(trajectory)
    for i in range(1, trajectory.shape[1]):
        delta_X[i, 0:3] = trajectory[i, 0:3] - trajectory[i-1, 0:3]
    return delta_X

def fill_with_blank(trajectory):
    if trajectory.shape[0] < MAX_TRAJ_STEPS:
        blank_vals = BLANK_VAL * np.ones((MAX_TRAJ_STEPS - trajectory.shape[0], EMBED_LOC), dtype=np.float64)
        trajectory = np.concatenate((trajectory, blank_vals), axis=0)
    return trajectory

def process_panda_to_model_input(trajectory):
    trajectory = np.array([np.array(trajectory[i]) for i in range(0, len(trajectory), ROBOT_DT)])
    trajectory = trajectory[0:MAX_TRAJ_STEPS, :]

    trajectory = translate(trajectory)
    trajectory = fill_with_blank(trajectory)
    trajectory = rotate_panda_to_match_orientation(trajectory)
    trajectory = round_down(trajectory)
    trajectory = delta_xyz(trajectory)
    return trajectory


def spilled(trajectory, properties=None):
    print("callling spilled...")
    trajectory = process_panda_to_model_input(trajectory)
    if properties is None:
        properties = np.array([SMALL_RADIUS_B, SMALL_HEIGHT, SMALL_RADIUS_U, SMALL_FILL_80])

    prediction = model.predict({"trajectory": trajectory[None, :, :],
                                "properties": properties[None, :],
                                })[0][0]
    
    print("prediction in spill-free: ", prediction, "spilled: ", prediction >= 0.5)
    return prediction >= 0.5
