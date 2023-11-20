import numpy as np
from process_data.process_data import process_data_SFC, process_data_PDM

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


def get_data_SFC(manual):
    X, Y = process_data_SFC()
    X_train, Y_train, X_val, Y_val = get_train_and_val_set(X, Y)
    return X_train, Y_train, X_val, Y_val, X, Y


def get_data_PDM(manual):
    X, Y = process_data_PDM()
    return X, Y


def get_data(model_name, manual=False):
    if model_name == "SFC":
        return get_data_SFC(manual=manual)
    elif model_name == "PDM":
        return get_data_PDM(manual=manual)
