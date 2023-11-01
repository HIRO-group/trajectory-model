import numpy as np
import matplotlib.pyplot as plt

from trajectory_model.data import get_data
from trajectory_model.spill_free.constants import MAX_TRAJ_STEPS, EMBED_DIM, MOCAP_DT, \
    BIG_RADIUS, BIG_HEIGHT, SMALL_RADIUS, SMALL_HEIGHT, \
    BIG_FILL_FULL, BIG_FILL_HALF, SMALL_FILL_FULL, SMALL_FILL_HALF, BLANK_VAL

X_train, Y_train, X_val, Y_val, X, Y = get_data(model_name='SFC', manual=False)



def keep_spill_free(X, Y):
    X_new, Y_new = [], []
    for e_id in range(X.shape[0]):
        if Y[e_id] == 0:
            X_new.append(X[e_id])
            Y_new.append(Y[e_id])
    X_new = np.array(X_new)
    Y_new = np.array(Y_new)
    return X_new, Y_new

def keep_selected_prop(X, Y):
    X_new, Y_new = [], []
    for e_id in range(X.shape[0]):
        prop = X[e_id, 0, 7:10]
        if prop[0] == BIG_RADIUS and prop[1] == BIG_HEIGHT and prop[2] == BIG_FILL_FULL:
            X_new.append(X[e_id])
            Y_new.append(Y[e_id])
    X_new = np.array(X_new)
    Y_new = np.array(Y_new)
    return X_new, Y_new

def keep_non_blanc(x_coord, y_coord):
    x_coord_new, y_coord_new = [], []
    for i in range(len(x_coord)):
        if x_coord[i] >= 10 or y_coord[i] >= 10 or \
            x_coord[i] <= -10 or y_coord[i] <= -10:
            continue
        else:
            x_coord_new.append(x_coord[i])
            y_coord_new.append(y_coord[i])
    return x_coord_new, y_coord_new


X_sf, Y_sf = keep_spill_free(X, Y)
X_sf, Y_sf = keep_selected_prop(X_sf, Y_sf)


x_coords = X_sf[:, :, 0].flatten()
y_coords = X_sf[:, :, 1].flatten()

x_coords, y_coords = keep_non_blanc(x_coords, y_coords)

# only keep the  x_coords and y_coords that are not blank


# Create a 2D histogram
heatmap, xedges, yedges = np.histogram2d(y_coords, x_coords, bins=(10,10))

# Plot the heatmap
import matplotlib.cm as cm

plt.imshow(heatmap.T, extent=[xedges.min(), xedges.max(), yedges.min(), yedges.max()],
            origin='lower', cmap=cm.RdYlGn)
plt.colorbar()
plt.title('Heatmap of (x, y) Coordinates')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')

plt.show()









