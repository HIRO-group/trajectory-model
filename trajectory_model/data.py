import numpy as np
import matplotlib.pyplot as plt

from process_data import process_data
from process_data_wp import process_data as process_data_wp

def get_data(spill_free=True):
    if spill_free:
        X, Y = process_data() 
    else:
        X, Y = process_data_wp()
        X = X[:, :, 0:3] # to get only positional data
        Y = Y[:, 0:3]

    train_id = int(3*X.shape[0]/4)
    train_index = np.random.choice(X.shape[0], train_id, replace=False) 
    val_index = np.setdiff1d(np.arange(X.shape[0]), train_index)
    X_train, Y_train = X[train_index], Y[train_index]
    X_val, Y_val = X[val_index], Y[val_index]

    # print("Train index: ", train_index)
    # print("val index: ", val_index)

    return X_train, Y_train, X_val, Y_val, X, Y
