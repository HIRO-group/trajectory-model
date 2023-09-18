import numpy as np 

from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, \
    Layer, Dense, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Model, Sequential
import tensorflow as tf


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


class TrajectoryClassifier(Model):
    def __init__(self, max_traj_steps, embed_dim, num_heads, ff_dim, dropout_rate=0.1, name="trajectory_classifier", **kwargs) -> None:
        super(TrajectoryClassifier, self).__init__(name=name, **kwargs)
        self.position_encoding = PositionalEnconding(max_traj_steps, embed_dim)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)
        self.pooling = GlobalAveragePooling1D()
        self.dropout1 = Dropout(dropout_rate)
        self.dense1 = Dense(20, activation="relu")
        self.dropout2 = Dropout(dropout_rate)
        self.dense2 = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.position_encoding(inputs)
        x = self.transformer_block(x)
        x = self.pooling(x)
        x = self.dropout1(x)
        x = self.dense1(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        return x
