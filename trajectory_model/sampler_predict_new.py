import numpy as np
import matplotlib.pyplot as plt

from process_data.process_data import keep_spill_free
from trajectory_model.data import get_data

from trajectory_model.spill_free.constants import \
    MAX_TRAJ_STEPS, EMBED_LOC, BLANK_VAL, ROBOT_DT, \
    BIG_DIAMETER_B, BIG_HEIGHT, BIG_DIAMETER_U, BIG_FILL_80, BIG_FILL_30, \
    SMALL_DIAMETER_B, SMALL_HEIGHT, SMALL_DIAMETER_U, SMALL_FILL_80, SMALL_FILL_50, \
    SHORT_TUMBLER_DIAMETER_B, SHORT_TUMBLER_HEIGHT, SHORT_TUMBLER_DIAMETER_U, SHORT_TUMBLER_FILL_30, SHORT_TUMBLER_FILL_70, \
    TALL_TUMBLER_DIAMETER_B, TALL_TUMBLER_HEIGHT, TALL_TUMBLER_DIAMETER_U, TALL_TUMBLER_FILL_50, TALL_TUMBLER_FILL_80, \
    TUMBLER_DIAMETER_B, TUMBLER_HEIGHT, TUMBLER_DIAMETER_U, TUMBLER_FILL_30, TUMBLER_FILL_70, \
    WINE_DIAMETER_B, WINE_HEIGHT, WINE_DIAMETER_U, WINE_FILL_30, WINE_FILL_70


def keep_selected_prop(X, Y, properties):
    X_new, Y_new = [], []
    for e_id in range(X.shape[0]):
        prop = X[e_id, 0, 7:]
        if prop[0] == properties[0] and \
              prop[1] == properties[1] and  \
                prop[2] == properties[2] and \
                    prop[3] == properties[3]:
            X_new.append(X[e_id])
            Y_new.append(Y[e_id])
    X_new = np.array(X_new)
    Y_new = np.array(Y_new)
    return X_new, Y_new


def convert_to_joint_angles(trajectory): # This is the whole trajectory starting from 0
    panda_start_joint_angles = [-0.412894, 0.841556, -1.21653, -1.45469, 0.766197, 1.79544, -0.893028]
    pass
    # joint_angles = []
    # for i in range(trajectory.shape[0]):
    #     loc = trajectory[i, 0:7]
    #     joint_angles.append(EMBED_LOC[loc[0], loc[1], loc[2], loc[3], loc[4], loc[5], loc[6]])
    # return np.array(joint_angles)

def get_probability_distribution_map(X):
    probability_distribution_map = []
    for e_id in range(X.shape[0]):
        trajectory = X[e_id, :, 0:7]
        trajectory = convert_to_joint_angles(trajectory)
        for joint_angles in trajectory:
            # xyz = loc[0:3]
            # if xyz[0] >= 10 or xyz[1] >= 10 or xyz[2] >= 10 or \
            #     xyz[0] <= -10 or xyz[1] <= -10 or xyz[2] <= -10:
            #     continue
            probability_distribution_map.append(joint_angles)
    return np.array(probability_distribution_map)

X, Y = get_data(model_name='PDM', manual=False)
properties = [BIG_DIAMETER_B, BIG_HEIGHT, BIG_DIAMETER_U, BIG_FILL_30]
X, Y = keep_selected_prop(X, Y, properties)
# probability_distribution_map = get_probability_distribution_map(X)

# def sample_state():
#     index = np.random.randint(0, len(probability_distribution_map))
#     return probability_distribution_map[index]
