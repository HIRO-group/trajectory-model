import numpy as np

import numpy as np
from scipy.spatial.transform import Rotation
import time

from trajectory_model.data import get_data
from trajectory_model.model import TrajectoryClassifier
from trajectory_model.constants import MAX_TRAJ_STEPS, EMBED_DIM, NUM_HEADS, FF_DIM, dt

max_traj_steps = 83
model = TrajectoryClassifier(max_traj_steps=max_traj_steps, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, ff_dim=FF_DIM)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.build((None, max_traj_steps, EMBED_DIM))
model.load_weights("/Users/ava/Documents/CU/Research/Repositories/HIRO/trajectory-model/weights/predict_class_real_data_latest.h5")
from trajectory_model.helper import plot_X


if __name__ == "__main__":
    path_prefix = "/Users/ava/Documents/CU/Research/Repositories/HIRO/trajectory-model/data/panda_ompl"
    path = f'{path_prefix}/data_0.9990000128746033_1688595649.4850183'
    X = np.loadtxt(path, delimiter=',')
    X = X.reshape(1, 83, 8)
    print(X[0, 0, :])
    # processed data: 
    # [ 0. 0. 0. 0.00484921 -0.0188662   0.035517, -0.99917924  1.] -> [ x: -0.4787902, y: 2.180395, z: -4.062466 ]
    # in visualize saved data: 
    # [ 0. 0. 0. -0.0033967 0.01908001 -0.03515668 0.99919389 1.] -> [ x: -0.312284, y: 2.1988694, z: -4.0242532 ]

    # I shouldn't use quaternion data: because it's not unique, 2 quaternions can represent the same orientation