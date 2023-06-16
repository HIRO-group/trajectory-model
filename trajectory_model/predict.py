import numpy as np

from trajectory_model.data import get_data
from trajectory_model.model import TrajectoryClassifier
from trajectory_model.constants import MAX_TRAJ_STEPS, EMBED_DIM, NUM_HEADS, FF_DIM, dt

max_traj_steps = 83
model = TrajectoryClassifier(max_traj_steps=max_traj_steps, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, ff_dim=FF_DIM)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.build((None, max_traj_steps, EMBED_DIM))
model.load_weights("weights/predict_class_real_data_83.h5")

def spilled(trajectory):
    # Trajectory is a array of joint angles = [[q1, ..., q7], [q1, ..., q7], ...]
    # Should perform FK
    # Should paramerize the trajectory by time
    # For now it's just gonna return something

    trajectory = np.zeros((1, 83, 8))
    return model.predict(trajectory) >= 0.5
