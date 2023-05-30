import numpy as np
from data import get_data
from model import TrajectoryClassifier
from constants import MAX_TRAJ_STEPS, EMBED_DIM, NUM_HEADS, FF_DIM, dt


if __name__ == '__main__':
    fit_model = False

    model = TrajectoryClassifier(max_traj_steps=MAX_TRAJ_STEPS, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, ff_dim=FF_DIM)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    x_train, y_train, x_val, y_val = get_data(data_num=2, max_traj_steps=MAX_TRAJ_STEPS, embed_dim=EMBED_DIM, dt=dt, debug=True)

    if fit_model:
        history = model.fit(x_train, y_train, 
                            batch_size=1, epochs=30, 
                            validation_data=(x_val, y_val)
                           )
        model.save_weights("weights/predict_class.h5")
    else:
        model.build((None, MAX_TRAJ_STEPS, EMBED_DIM))
        model.load_weights("weights/predict_class.h5")

    print('data 0 is supposed to be slosh free (<0.5): ', model.predict(x_train[np.newaxis, 0]))
    print('data 1 is supposed to spill (>=0.5): ', model.predict(x_train[np.newaxis, 1]))

