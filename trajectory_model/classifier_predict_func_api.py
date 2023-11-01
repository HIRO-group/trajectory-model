import numpy as np
from trajectory_model.spill_free.model_func_api import get_SFC_model
from trajectory_model.spill_free.constants import \
      MAX_TRAJ_STEPS, BIG_RADIUS, BIG_HEIGHT, SMALL_RADIUS, \
      SMALL_HEIGHT, BIG_FILL_FULL, BIG_FILL_HALF, \
      SMALL_FILL_FULL, SMALL_FILL_HALF, ROBOT_DT, BLANK_VAL, EMBED_LOC
from trajectory_model.helper.rotate_quaternion import rotate_panda_to_match_orientation

# from trajectory_model.helper.helper import quat_to_euler, euler_to_quat, ctime_str
# from trajectory_model.helper.rotate_quaternion import quaternion_to_angle_axis, rotate_quaternion

model = get_SFC_model()
model.load_weights("/home/ava/projects/trajectory-model/weights/spill_classifier_func_api/best/2023-10-31 20:11:14_epoch_169_train_acc_0.91.h5")

def translate(trajectory):
    xyz = trajectory[:, 0:3] # shape: (T, 3)
    xyz = xyz - trajectory[0, 0:3] # shape: (T, 3)
    trajectory[:, 0:3] = xyz
    return trajectory

def spilled(trajectory):
    trajectory = np.array([np.array(trajectory[i]) for i in range(0, len(trajectory), ROBOT_DT)])
    trajectory = trajectory[0:MAX_TRAJ_STEPS, :]
    trajectory = translate(trajectory)

    if trajectory.shape[0] < MAX_TRAJ_STEPS:
        blank_vals = BLANK_VAL * np.ones((MAX_TRAJ_STEPS - trajectory.shape[0], EMBED_LOC), dtype=np.float64)
        trajectory = np.concatenate((trajectory, blank_vals), axis=0)
    
    trajectory = rotate_panda_to_match_orientation(trajectory)
    
    properties = np.array([BIG_RADIUS, BIG_HEIGHT, BIG_FILL_HALF])
    prediction = model.predict({"trajectory": trajectory[None, :, :],
                                "properties": properties[None, :],
                                })[0][0]
    
    print("prediction in spill-free: ", prediction, "spilled: ", prediction >= 0.5)