import numpy as np
from trajectory_model.spill_free.model_func_api import get_SFC_model
from trajectory_model.spill_free.constants import \
      MAX_TRAJ_STEPS, BIG_RADIUS, BIG_HEIGHT, SMALL_RADIUS, \
      SMALL_HEIGHT, BIG_FILL_FULL, BIG_FILL_HALF, \
      SMALL_FILL_FULL, SMALL_FILL_HALF, ROBOT_DT, BLANK_VAL, EMBED_LOC
from trajectory_model.helper.rotate_quaternion import rotate_panda_to_match_orientation

model = get_SFC_model()
# address = "/home/ava/projects/trajectory-model/weights/spill_classifier_func_api/best/2023-11-08 15:34:11_epoch_254_train_acc_0.88.h5"
# address = "/home/ava/projects/trajectory-model/weights/spill_classifier_func_api/best/2023-11-08 18:02:34_epoch_292_train_acc_0.89.h5"
# address = "/home/ava/projects/trajectory-model/weights/spill_classifier_func_api/best/2023-11-13 13:45:23_epoch_262_train_acc_0.91.h5"
# address = "/home/ava/projects/trajectory-model/weights/spill_classifier_func_api/best/2023-11-13 16:39:00_epoch_297_train_acc_0.9.h5"
# address = "/home/ava/projects/trajectory-model/weights/spill_classifier_func_api/best/2023-11-13 16:56:43_epoch_288_train_acc_0.89.h5"
# address = '/home/ava/projects/trajectory-model/weights/spill_classifier_func_api/best/2023-11-13 17:17:50_epoch_298_train_acc_0.91.h5'
address = '/home/ava/projects/trajectory-model/weights/spill_classifier_func_api/best/2023-11-13 17:39:44_epoch_298_train_acc_0.89.h5'
model.load_weights(address)


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
        properties = np.array([BIG_RADIUS, BIG_HEIGHT, BIG_FILL_HALF])

    prediction = model.predict({"trajectory": trajectory[None, :, :],
                                "properties": properties[None, :],
                                })[0][0]
    
    print("prediction in spill-free: ", prediction, "spilled: ", prediction >= 0.5)
    return prediction >= 0.5
