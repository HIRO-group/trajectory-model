import numpy as np 

from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, \
    Layer, Dense, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Model, Sequential
import tensorflow as tf
from keras import layers
from tensorflow import keras

from trajectory_model.spill_free.constants import MAX_TRAJ_STEPS, EMBED_LOC, EMBED_PROP

# from trajectory_model.con

class PositionalEnconding(Layer):
    def __init__(self, max_traj_steps, embed_dim) -> None:
        super(PositionalEnconding, self).__init__()
        self.max_traj_steps = max_traj_steps
        self.embed_dim = embed_dim
    
    def call(self, inputs):
        pos_encoding = self.positional_encoding(self.max_traj_steps, self.embed_dim)
        inputs = inputs + pos_encoding[:, :, :]
        return inputs

    def positional_encoding(self, positions, d): 
        angle_rads = self.get_angles(np.arange(positions)[:, np.newaxis],
                                np.arange(d)[np.newaxis, :],
                                d)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, k, d):
        i = k // 2
        angles = pos/(np.power(10000,2*i/d))
        return angles



class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"), 
             Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def get_SFC_model():
    trajectory_input = keras.Input(shape=(MAX_TRAJ_STEPS, EMBED_LOC), name="trajectory")
    properties_input = keras.Input(shape=(EMBED_PROP,), name="properties")
    # properties_features = layers.Embedding(10, 5)(properties_input) # 10 is the vocab size, 5 is the embedding dimension

    traj_pos_enc = PositionalEnconding(MAX_TRAJ_STEPS, EMBED_LOC)(trajectory_input)
    traj_features = tf.keras.layers.Add()([trajectory_input, traj_pos_enc[: , :tf.shape(trajectory_input)[1], :] ])
    traj_features = TransformerBlock(EMBED_LOC, num_heads=8, ff_dim=32, dropout_rate=0.1)(traj_features)
    traj_features = GlobalAveragePooling1D()(traj_features)
    traj_features = Dropout(0.1)(traj_features)

    x = layers.concatenate([traj_features, properties_input])

    x = Dense(20, activation="relu")(x)
    x = Dropout(0.1)(x)
    prediction = Dense(1, activation='sigmoid', name="prediction")(x)

    model = keras.Model(
        inputs=[trajectory_input, properties_input],
        outputs=[prediction],
    )

    return model
