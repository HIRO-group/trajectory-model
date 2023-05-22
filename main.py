import numpy as np
from data import get_data, process_data, MAX_TRAJ_STEPS, EMBED_DIM
from layer import TransformerBlock

from tensorflow.keras.layers import Input, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf

NUM_HEADS = 8
FF_DIM = 32
transformer_block = TransformerBlock(EMBED_DIM, NUM_HEADS, FF_DIM)

inputs = Input(shape=(MAX_TRAJ_STEPS, EMBED_DIM))

x = transformer_block(inputs)
x = GlobalAveragePooling1D()(x)
x = Dropout(0.1)(x)
x = Dense(20, activation="relu")(x)
x = Dropout(0.1)(x)
outputs = Dense(1, activation='sigmoid')(x)
model = Model(inputs=inputs, outputs=outputs)


x_train_raw, y_train, x_val_raw, y_val = get_data(debug=True)

x_train, x_val = process_data(x_train_raw, x_val_raw)



model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

history = model.fit(x_train, y_train, 
                    batch_size=1, epochs=30, 
                    validation_data=(x_val, y_val)
                   )
# model.save_weights("weights/predict_class.h5")

model.load_weights("weights/predict_class.h5")
# model.evaluate(x_val, y_val)
# loss, acc, = model.evaluate(x_val, y_val)
# print("Dev set accuracy = ", acc)
# print(x_val.shape)
# print(x_train[np.newaxis, 0].shape)
print('data0: ', model.predict(x_train[np.newaxis, 0]))
print('data1: ', model.predict(x_train[np.newaxis, 1]))

