import numpy as np
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, \
    Layer, Dense, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Model, Sequential
import tensorflow as tf
import os
import random

# seed_value= 0

# np.random.seed(seed_value)
# tf.random.set_seed(seed_value)
# random.seed(seed_value)
# os.environ['PYTHONHASHSEED']=str(seed_value)

class PositionalEnconding(Layer):
    def __init__(self, max_traj_steps, embed_dim) -> None:
        super(PositionalEnconding, self).__init__()
        self.max_traj_steps = max_traj_steps
        self.embed_dim = embed_dim

    def call(self, inputs):
        pos_encoding = self.positional_encoding(
            self.max_traj_steps, self.embed_dim)
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
        angles = pos/(np.power(10000, 2*i/d))
        return angles


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, activation_func="relu"):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [Dense(ff_dim, activation=activation_func),
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


class PositionSampler(Model):
    def __init__(self, embed_X, embed_Y, max_num_waypoints, name="PositionSampler", **kwargs) -> None:
        super(PositionSampler, self).__init__(name=name, **kwargs)

        num_heads, ff_dim = 8, 8
        tf_block_dropout = 0.1
        dropout1 = 0.05
        dropout2 = 0.05

        self.position_encoding = PositionalEnconding(max_num_waypoints, embed_X)
        self.transformer_block = TransformerBlock(embed_X, num_heads=num_heads,
                                                  ff_dim=ff_dim, dropout_rate=tf_block_dropout,
                                                  activation_func="relu")
        # self.layernorm = LayerNormalization(epsilon=1e-6)
        # self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.pooling = GlobalAveragePooling1D()
        self.dropout1 = Dropout(dropout1)
        self.dense1 = Dense(embed_Y, activation="linear")

        self.dropout2 = Dropout(dropout2)
        self.dense2 = Dense(embed_Y, activation="linear")

        self.dropout3 = Dropout(dropout2)
        self.dense3 = Dense(embed_Y, activation="linear")

    def call(self, inputs):
        x = self.position_encoding(inputs)

        x = self.transformer_block(x)
        # x = self.transformer_block(x)
        # x = self.transformer_block(x)
        # x = self.transformer_block(x)

        # x = self.transformer_block(x)
        # x = self.transformer_block(x)

        # x = self.layernorm(x)
        x = self.pooling(x)
        x = self.dropout1(x)
        # x = self.layernorm2(x)
        x = self.dense1(x)

        x = self.dropout2(x)
        x = self.dense2(x)

        # x = self.dropout3(x)
        # x = self.dense3(x)

        return x


class OrientationSampler(Model):
    def __init__(self, embed_X, embed_Y, max_num_waypoints, name="OrientationSampler", **kwargs) -> None:
        super(OrientationSampler, self).__init__(name=name, **kwargs)
        
        num_heads, ff_dim = 8, 8
        tf_block_dropout = 0.1
        dropout1 = 0.05
        dropout2 = 0.05
        
        self.position_encoding = PositionalEnconding(max_num_waypoints, embed_X)
        self.transformer_block = TransformerBlock(embed_X, num_heads=num_heads,
                                                  ff_dim=ff_dim, dropout_rate=tf_block_dropout,
                                                  activation_func="relu")
        self.pooling = GlobalAveragePooling1D()
        self.dropout1 = Dropout(dropout1)
        self.dense1 = Dense(embed_Y, activation="tanh")
        self.dropout2 = Dropout(dropout2)
        self.dense2 = Dense(embed_Y, activation="linear")
    

    def call(self, inputs):
        x = self.position_encoding(inputs)
        x = self.transformer_block(x)
        x = self.pooling(x)
        x = self.dropout1(x)
        x = self.dense1(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        return x