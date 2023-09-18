from constants import MAX_TRAJ_STEPS, EMBED_DIM, NUM_HEADS, FF_DIM, dt
import numpy as np
from model import TrajectoryClassifier
from data import read_panda_data

max_traj_steps = 2000

model = TrajectoryClassifier(max_traj_steps=max_traj_steps, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, ff_dim=FF_DIM)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# model.build((None, max_traj_steps, EMBED_DIM))
# model.load_weights("weights/predict_class_panda_real_data.h5")

X_train, Y_train, X_val, Y_val, max_traj_steps, X, Y = read_panda_data()
history = model.fit(X_train, Y_train, 
                    batch_size=1, epochs=10, 
                    validation_data=(X_val, Y_val)
                    )
# model.save_weights("weights/predict_class_panda_real_data.h5")


slosh_free = model.predict(X_val[np.newaxis, 0])
print("should be slosh_free: ", slosh_free)