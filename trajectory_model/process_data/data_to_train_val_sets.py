import numpy as np
from trajectory_model.process_data.data_processor import process_data

def get_manual_train_and_val_index(X):
    train_id = int(3*X.shape[0]/4)
    train_index = np.arange(train_id)
    val_index = np.arange(train_id, X.shape[0])
    return train_index, val_index

def get_train_and_val_set(X, Y, manual=False):
    train_id = int(3*X.shape[0]/4)
    if manual:
        train_index, val_index = get_manual_train_and_val_index(X)
    else:
        train_index = np.random.choice(X.shape[0], train_id, replace=False) 
        val_index = np.setdiff1d(np.arange(X.shape[0]), train_index)
    
    X_train, Y_train = X[train_index], Y[train_index]
    X_val, Y_val = X[val_index], Y[val_index]
    return X_train, Y_train, X_val, Y_val


def get_X_and_Y():
    X, Y = process_data()
    X_train, Y_train, X_val, Y_val = get_train_and_val_set(X, Y)    
    X_train_traj = X_train[:, :, :7]
    X_train_prop = X_train[:, 0, 7:]
    X_val_traj = X_val[:, :, :7]
    X_val_prop = X_val[:, 0, 7:]

    return (X_train_traj, X_train_prop, X_val_traj, X_val_prop), (Y_train, Y_val)


def get_train_and_val_sets(X, Y):
    X_train_traj, X_train_prop, X_val_traj, X_val_prop = X[0], X[1], X[2], X[3]
    Y_train, Y_val = Y[0], Y[1]
    return X_train_traj, X_train_prop, Y_train, X_val_traj, X_val_prop, Y_val
    