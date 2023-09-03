import numpy as np
from trajectory_model.spill_free.model import TrajectoryClassifier
from trajectory_model.spill_free.constants import EMBED_DIM, NUM_HEADS, FF_DIM, MAX_TRAJ_STEPS
from trajectory_model.helper import quat_to_euler, euler_to_quat, ctime_str

model = TrajectoryClassifier(max_traj_steps=MAX_TRAJ_STEPS, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, ff_dim=FF_DIM)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.build((None, MAX_TRAJ_STEPS, EMBED_DIM))
model.load_weights("/home/ava/projects/trajectory-model/weights/spill_classifier/acc_0.9_loss_0.32_data_num_186_epochs_80.h5")

def convert_to_model_input(trajectory):
    step_size = int(len(trajectory) / MAX_TRAJ_STEPS)
    remainder = len(trajectory) % MAX_TRAJ_STEPS
    trajectory = np.array([np.array(trajectory[i]) for i in range(0, len(trajectory) - remainder, step_size)])
    trajectory = np.append(trajectory, np.ones((trajectory.shape[0], 1)), axis=1)
    trajectory = trajectory.reshape(1, MAX_TRAJ_STEPS, EMBED_DIM)
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
    name = f'data_{round(prediction, 2)}_{ctime_str()}'
    path = f'/home/ava/projects/trajectory-model/data/panda_ompl/{name}'
    trajectory = trajectory.reshape(MAX_TRAJ_STEPS, EMBED_DIM)
    np.savetxt(path, trajectory, delimiter=',')
    print(f'successfully saved to file: {name}')


def spilled(trajectory):
    trajectory = convert_to_model_input(trajectory)
    trajectory = transform_trajectory(trajectory)
    prediction = model.predict(trajectory)[0][0]   
    # print("Prediction in python function:", prediction)
    # save_trajectory(trajectory, prediction)
    return prediction
 