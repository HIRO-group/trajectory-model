import numpy as np
import matplotlib.pyplot as plt

from process_data import process_data
from process_data_wp import process_data as process_data_wp


def get_manual_train_and_val_index():
    val_index = [1,4,5,6,7,17,18,21,22,23,24,30,32,33,42,48,53,55,59,61,67,68,74,77,78,89,90,98,106,111,119,125,132,134,135,137,148,150,154,155,156,159,170,172,177,182,187,188,194,201,202,203,206,207,209]
    train_index = [97,165,26,10,116,3,147,197,139,179,46,121,216,214,195,103,145,126,104,178,0,122,167,131,16,218,151,184,171,80,31,14,62,168,143,9,13,158,94,19,199,58,65,196,110,57,142,92,105,162,84,87,79,56,83,169,107,99,140,120,118,28,25,173,114,117,85,34,93,211,183,136,186,29,41,96,185,205,141,198,38,189,45,217,37,35,52,49,40,108,95,129,109,27,64,210,20,181,124,192,212,193,54,146,144,70,127,175,15,102,71,157,47,50,190,180,2,128,213,153,204,174,133,86,12,11,176,91,82,152,63,75,112,51,76,115,44,88,81,60,161,101,166,160,208,130,43,100,191,149,113,73,138,123,164,69,200,72,219,163,215,66,8,36,39]
    print("len val_index, len train_index: ", len(val_index), len(train_index))
    return train_index, val_index

def get_train_and_val_set(X, Y, manual=False):
    train_id = int(3*X.shape[0]/4)
    if manual:
        print("Getting manual train and val index...")
        train_index, val_index = get_manual_train_and_val_index()
    else:
        train_index = np.random.choice(X.shape[0], train_id, replace=False) 
        val_index = np.setdiff1d(np.arange(X.shape[0]), train_index)
    X_train, Y_train = X[train_index], Y[train_index]
    X_val, Y_val = X[val_index], Y[val_index]
    # print("val_index: ", val_index)
    # print("train_index: ", train_index)
    return X_train, Y_train, X_val, Y_val


def get_position_wp_data(manual=False):
    X_wp, Y_wp = process_data_wp()
    Y = Y_wp[:, 0:3]
    X_pos = X_wp[:, :, 0:3] # to get only positional data
    X_cup = X_wp[:, :, 7:8] # to get only cup data
    X = np.concatenate((X_pos, X_cup), axis=2)

    mean_X = np.mean(X, axis=(0, 1), keepdims=True)
    std_X = np.std(X, axis=(0, 1), keepdims=True)
    mean_Y = np.mean(Y, axis=0, keepdims=True)
    std_Y = np.std(Y, axis=0, keepdims=True)

    X = (X - mean_X) / std_X
    Y = (Y - mean_Y) / std_Y

    X_train, Y_train, X_val, Y_val = get_train_and_val_set(X, Y, manual=manual)
    return X_train, Y_train, X_val, Y_val, X, Y


def get_orientation_wp_data(manual=False):
    X_wp, Y_wp = process_data_wp()

    # X should be (pos, a, b, c, w, cup_type) for now
    # Y should be (a, b, c, w)
    
    X = X_wp[:, :, :]
    Y = Y_wp[:, 3:7]

    mean_X = np.mean(X, axis=(0, 1), keepdims=True)
    std_X = np.std(X, axis=(0, 1), keepdims=True)
    mean_Y = np.mean(Y, axis=0, keepdims=True)
    std_Y = np.std(Y, axis=0, keepdims=True)

    X = (X - mean_X) / std_X
    Y = (Y - mean_Y) / std_Y
    
    X_train, Y_train, X_val, Y_val = get_train_and_val_set(X, Y, manual=manual)

    return X_train, Y_train, X_val, Y_val, X, Y


def get_data():
    X, Y = process_data()
    X_train, Y_train, X_val, Y_val = get_train_and_val_set(X, Y)
    return X_train, Y_train, X_val, Y_val, X, Y
