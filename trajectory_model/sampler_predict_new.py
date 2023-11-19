import numpy as np
import matplotlib.pyplot as plt

from process_data.process_data_SFC import keep_spill_free
from trajectory_model.data import get_data

from trajectory_model.spill_free.constants import \
    MAX_TRAJ_STEPS, EMBED_LOC, BLANK_VAL, ROBOT_DT, \
    BIG_RADIUS_B, BIG_HEIGHT, BIG_RADIUS_U, BIG_FILL_80, BIG_FILL_30, \
    SMALL_RADIUS_B, SMALL_HEIGHT, SMALL_RADIUS_U, SMALL_FILL_80, SMALL_FILL_50, \
    SHORT_TUMBLER_RADIUS_B, SHORT_TUMBLER_HEIGHT, SHORT_TUMBLER_RADIUS_U, SHORT_TUMBLER_FILL_30, SHORT_TUMBLER_FILL_70, \
    TALL_TUMBLER_RADIUS_B, TALL_TUMBLER_HEIGHT, TALL_TUMBLER_RADIUS_U, TALL_TUMBLER_FILL_50, TALL_TUMBLER_FILL_80, \
    TUMBLER_RADIUS_B, TUMBLER_HEIGHT, TUMBLER_RADIUS_U, TUMBLER_FILL_30, TUMBLER_FILL_70, \
    WINE_RADIUS_B, WINE_HEIGHT, WINE_RADIUS_U, WINE_FILL_30, WINE_FILL_70


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

def get_probability_distribution_map(X):
    probability_distribution_map = []
    for e_id in range(X.shape[0]):
        trajectory = X[e_id, :, 0:7]
        for loc in trajectory:
            xyz = loc[0:3]
            if xyz[0] >= 10 or xyz[1] >= 10 or xyz[2] >= 10 or \
                xyz[0] <= -10 or xyz[1] <= -10 or xyz[2] <= -10:
                continue
            probability_distribution_map.append(loc)
    return np.array(probability_distribution_map)

_, _, _, _, X, Y = get_data(model_name='SFC', manual=False)
properties = [BIG_RADIUS_B, BIG_HEIGHT, BIG_RADIUS_U, BIG_FILL_30]
X, Y = keep_spill_free(X, Y)
X, Y = keep_selected_prop(X, Y, properties)
probability_distribution_map = get_probability_distribution_map(X)

def sample_state():
    index = np.random.randint(0, len(probability_distribution_map))
    # convert to robot coordinate
    return probability_distribution_map[index]
