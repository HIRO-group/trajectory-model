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
model.load_weights("/home/ava/npm/trajectory-model/weights/predict_class_real_data_rn.h5")

def quat_to_euler(quaternions):
    rot = Rotation.from_quat(quaternions)
    rot_euler = rot.as_euler('xyz', degrees=True)
    return rot_euler

def euler_to_quat(euler_angles):
    rot = Rotation.from_euler('xyz', euler_angles, degrees=True)
    rot_quat = rot.as_quat()
    return rot_quat

def convert_to_model_input(trajectory):
    step_size = int(len(trajectory) / 83)
    remainder = len(trajectory) % 83
    trajectory = np.array([np.array(trajectory[i]) for i in range(0, len(trajectory) - remainder, step_size)])
    trajectory = np.append(trajectory, np.ones((trajectory.shape[0], 1)), axis=1)
    trajectory = trajectory.reshape(1, 83, 8)
    return trajectory


def translate(trajectory):
    xyz = trajectory[0, :, 0:3] # shape: (T, 3)
    xyz = xyz - trajectory[0, 0, 0:3] # shape: (T, 3)    
    trajectory[0, :, 0:3] = xyz
    return trajectory

def rotate(trajectory):
    abcw = trajectory[0, :, 3:7] # shape: (T, 4)    
    phi_theta_psi = quat_to_euler(abcw) # shape: (T, 3)
    phi_theta_psi = phi_theta_psi - phi_theta_psi[0] + np.array(([-0.4661262, 2.1714807, -4.0390808])) # shape: (T, 3)
    abcw = euler_to_quat(phi_theta_psi) # shape: (T, 4)
    trajectory[0, :, 3:7] = abcw
    return trajectory    


def transform_trajectory(trajectory):
    trajectory = translate(trajectory)
    trajectory = rotate(trajectory)
    return trajectory


def save_trajectory(trajectory, prediction):
    path = f'/home/ava/npm/trajectory-model/data/panda_ompl/data_{round(prediction, 3)}_{time.time()}'
    trajectory = trajectory.reshape(83, 8)
    np.savetxt(path, trajectory, delimiter=',')
    print("successfully saved to file")


def spilled(trajectory):
    trajectory = convert_to_model_input(trajectory)
    trajectory = transform_trajectory(trajectory)
    prediction = model.predict(trajectory)[0][0]    
    print("Prediction in python function:", prediction)
    save_trajectory(trajectory, prediction)
    return prediction
 