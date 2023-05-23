import numpy as np
from data import get_data
from model import TrajectoryClassifier

T_max = 3
dt = 0.02
MAX_TRAJ_STEPS = int(T_max//dt)
NUM_HEADS = 8
FF_DIM = 32
EMBED_DIM = 16
''' 
embedding = (x, xdot, xddot, o, odot, cup_type) 
x=(x, y, z)
o=(phi, thetha, psi)
cup_type = 1 or 2 or 3
'''

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

