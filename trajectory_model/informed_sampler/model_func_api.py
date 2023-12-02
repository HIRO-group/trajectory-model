import numpy as np 
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, \
    Layer, Dense, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Model, Sequential
import tensorflow as tf
from keras import layers
from tensorflow import keras

from trajectory_model.spill_free.constants import MAX_TRAJ_STEPS, EMBED_LOC, EMBED_PROP


def get_SFS_model():
    properties_input = keras.Input(shape=(EMBED_PROP,), name="properties")
    x = Dense(20, activation="sigmoid")(properties_input)
    x = Dropout(0.1)(x)
    x = Dense(5, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(100, activation='sigmoid', name="prediction")(x)
    pdm = keras.layers.Reshape((10, 10, 1))(x)
    model = keras.Model(
        inputs=[properties_input],
        outputs=[pdm],
    )
    return model

sfs = get_SFS_model()
sfs.summary()