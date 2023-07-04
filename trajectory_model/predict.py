import numpy as np
from scipy.spatial.transform import Rotation

from trajectory_model.data import get_data
from trajectory_model.model import TrajectoryClassifier
from trajectory_model.constants import MAX_TRAJ_STEPS, EMBED_DIM, NUM_HEADS, FF_DIM, dt

max_traj_steps = 83
model = TrajectoryClassifier(max_traj_steps=max_traj_steps, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, ff_dim=FF_DIM)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.build((None, max_traj_steps, EMBED_DIM))
model.load_weights("weights/predict_class_real_data_83.h5")

def quat_to_euler(quaternions):
    rot = Rotation.from_quat(quaternions)
    rot_euler = rot.as_euler('xyz', degrees=True)
    return rot_euler

def euler_to_quat(euler_angles):
    rot = Rotation.from_euler('xyz', euler_angles, degrees=True)
    rot_quat = rot.as_quat()
    return rot_quat

def spilled(trajectory):
    trajectory = np.zeros((1, 83, 8))

    xyz = trajectory[0, :, 0:3] # shape: (T, 3)
    xyz = xyz - trajectory[0, 0, 0:3] # shape: (T, 3)
    abcw = trajectory[0, :, 3:7] # shape: (T, 4)
    phi_theta_psi = quat_to_euler(abcw) # shape: (T, 3)
    phi_theta_psi = phi_theta_psi - phi_theta_psi[0] + np.array(([-0.4661262, 2.1714807, -4.0390808])) # shape: (T, 3)
    abcw = euler_to_quat(phi_theta_psi) # shape: (T, 4)

    trajectory[0, :, 0:3] = xyz

    return model.predict(trajectory) >= 0.5 
