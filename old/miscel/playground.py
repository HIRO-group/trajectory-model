import numpy as np

import numpy as np
from scipy.spatial.transform import Rotation
import time

from trajectory_model.helper.helper import plot_X
from trajectory_model.spill_free.model import TrajectoryClassifier
from trajectory_model.spill_free.constants import EMBED_DIM, NUM_HEADS, FF_DIM, MAX_TRAJ_STEPS

model = TrajectoryClassifier(max_traj_steps=MAX_TRAJ_STEPS, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, ff_dim=FF_DIM)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.build((None, MAX_TRAJ_STEPS, EMBED_DIM))
model.load_weights("/Users/ava/Documents/CU/Research/Repositories/HIRO/trajectory-model/weights/predict_class_real_data_latest.h5")


if __name__ == "__main__":
    path_prefix = "/Users/ava/Documents/CU/Research/Repositories/HIRO/trajectory-model/data/panda_ompl"
    path = f'{path_prefix}/1'
    X = np.loadtxt(path, delimiter=',')
    X = X.reshape(1, MAX_TRAJ_STEPS, EMBED_DIM)
    # print(X[0, 0, :])

    plot_X(X, 0, 0.1)
    prediction = model.predict(X)[0][0]
    print("Probability of spilling: ", prediction)
    
    # processed data: 
    # [ 0. 0. 0. 0.00484921 -0.0188662   0.035517, -0.99917924  1.] -> [ x: -0.4787902, y: 2.180395, z: -4.062466 ]
    # in visualize saved data: 
    # [ 0. 0. 0. -0.0033967 0.01908001 -0.03515668 0.99919389 1.] -> [ x: -0.312284, y: 2.1988694, z: -4.0242532 ]

    