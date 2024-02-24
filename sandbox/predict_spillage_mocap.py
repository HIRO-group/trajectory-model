import numpy as np
from trajectory_model.classifier_predict_func_api import model as new_model
from trajectory_model.classifier_predict import model as old_model
from trajectory_model.helper.read import read_panda_vectors
from trajectory_model.spill_free.constants import MAX_TRAJ_STEPS, EMBED_DIM, MOCAP_DT, \
    BIG_HEIGHT, SMALL_HEIGHT, BLANK_VAL, BEAKER, CYLINDER, FLASK
from process_data.process_data import read_a_directory, fill_with_blanks, transform_trajectory, \
    round_down_orientation_and_pos, reverse_y_axis, compute_delta_X

def read_chemistry_vessel_trajectories():
    X = np.zeros((0, MAX_TRAJ_STEPS, EMBED_DIM), dtype=np.float64)
    Y = np.zeros((0, 1))
    dir_prefix = '/home/ava/projects/trajectory-model/data/generalization/'
    file_names = [
        # ('beaker/50', BEAKER, BEAKER.fill_level_50),
        # ('beaker/90', BEAKER, BEAKER.fill_level_90),
        # ('cylinder/20', CYLINDER, CYLINDER.fill_level_20),
        # ('cylinder/80', CYLINDER, CYLINDER.fill_level_80),
        ('flask/60', FLASK, FLASK.fill_level_60),
        ('flask/90', FLASK, FLASK.fill_level_90),
    ]
    for file in file_names:
        vessel = file[1]
        fill_level = file[2]

        directory_path_ns = dir_prefix + file[0] + '/spill-free/'
        X_ns = read_a_directory(directory_path_ns, vessel.diameter_b, vessel.height, vessel.diameter_u, fill_level)
        Y_ns = np.zeros((X_ns.shape[0], 1))
        X = np.concatenate((X, X_ns), axis=0)
        Y = np.concatenate((Y, Y_ns), axis=0)

        directory_path_s = dir_prefix + file[0] + '/spilled/'
        X_s = read_a_directory(directory_path_s, vessel.diameter_b, vessel.height, vessel.diameter_u, fill_level)
        Y_s = np.ones((X_s.shape[0], 1))
        X = np.concatenate((X, X_s), axis=0)
        Y = np.concatenate((Y, Y_s), axis=0)
    
    return X, Y


def predict_spillage(X, Y):
    correct, false = 0, 0
    for index in range(X.shape[0]):
        prediction = new_model.predict({"trajectory": X[index, :, :7][None, :, :],
                                    "properties": X[index, 0, 7:][None, :],
                                    })[0][0]
        # prediction= old_model.predict(X[index][None, :, :])[0][0]

        actual_value = Y[index]
        prediction = 0 if prediction < 0.5 else 1
        
        if prediction == actual_value:
            correct += 1
        else:
            false += 1
    
    print("correct: ", correct)
    print("false: ", false)
    print("accuracy: ", correct/(correct+false))


if __name__ == "__main__":
    X, Y = read_chemistry_vessel_trajectories()
    X = fill_with_blanks(X)
    X = transform_trajectory(X)
    X = round_down_orientation_and_pos(X)
    X = reverse_y_axis(X)
    X = compute_delta_X(X)
    predict_spillage(X, Y)
    

    
