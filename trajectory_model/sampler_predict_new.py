import numpy as np
import matplotlib.pyplot as plt

from process_data.process_data_SFC import keep_spill_free
from trajectory_model.data import get_data
from trajectory_model.spill_free.constants import MAX_TRAJ_STEPS, EMBED_DIM, MOCAP_DT, \
    BIG_RADIUS, BIG_HEIGHT, SMALL_RADIUS, SMALL_HEIGHT, \
    BIG_FILL_FULL, BIG_FILL_HALF, SMALL_FILL_FULL, SMALL_FILL_HALF, BLANK_VAL

def keep_selected_prop(X, Y, properties):
    X_new, Y_new = [], []
    for e_id in range(X.shape[0]):
        prop = X[e_id, 0, 7:10]
        if prop[0] == properties[0] and \
              prop[1] == properties[1] and  \
                prop[2] == properties[2]:
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
properties = [BIG_RADIUS, BIG_HEIGHT, BIG_FILL_HALF] # radius, height, fill-level
X, Y = keep_spill_free(X, Y)
X, Y = keep_selected_prop(X, Y, properties)
probability_distribution_map = get_probability_distribution_map(X)

def sample_state():
    index = np.random.randint(0, len(probability_distribution_map))
    # convert to robot coordinate
    return probability_distribution_map[index]
