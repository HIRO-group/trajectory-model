import numpy as np
from trajectory_model.SFC.constants import MAX_TRAJ_STEPS, EMBED_LOC, BLANK_VAL, ROBOT_DT
from trajectory_model.common.rotate_quaternion import rotate_panda_to_match_orientation

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
