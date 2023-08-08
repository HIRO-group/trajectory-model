import numpy as np
import matplotlib.pyplot as plt

from process_data import process_data
from process_data_wp import process_data as process_data_wp


def get_train_and_val_set(X, Y):
    train_id = int(3*X.shape[0]/4)
    train_index = np.random.choice(X.shape[0], train_id, replace=False) 
    val_index = np.setdiff1d(np.arange(X.shape[0]), train_index)
    X_train, Y_train = X[train_index], Y[train_index]
    X_val, Y_val = X[val_index], Y[val_index]
    return X_train, Y_train, X_val, Y_val


def get_position_wp_data():
    X_wp, Y_wp = process_data_wp()
    Y = Y_wp[:, 0:3]
    X_pos = X_wp[:, :, 0:3] # to get only positional data
    X_cup = X_wp[:, :, 7:8] # to get only cup data
    X = np.concatenate((X_pos, X_cup), axis=2)
    X_train, Y_train, X_val, Y_val = get_train_and_val_set(X, Y)
    return X_train, Y_train, X_val, Y_val, X, Y

def get_data():
    X, Y = process_data()
    X_train, Y_train, X_val, Y_val = get_train_and_val_set(X, Y)
    return X_train, Y_train, X_val, Y_val, X, Y
