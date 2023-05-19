import numpy as np

MAX_TRAJECTORY_LENGHT = 500
TRAJECTORY_FEATURES_NUM = 15

def get_data(debug=False):
    if debug:
        x_train = np.random.uniform(5, size=(MAX_TRAJECTORY_LENGHT, TRAJECTORY_FEATURES_NUM))
        y_train = np.random.randint(1)
        x_val = np.random.uniform(5, size=(MAX_TRAJECTORY_LENGHT, TRAJECTORY_FEATURES_NUM))
        y_val = np.random.randint(1)
        return x_train, y_train, x_val, y_val
    else:
        #have to make sure that the data is between a certain range so positional encoding works properly
        raise NotImplementedError
        
