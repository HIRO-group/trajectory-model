import numpy as np
from data import get_data
from model import TrajectoryClassifier
from constants import MAX_TRAJ_STEPS, EMBED_DIM, NUM_HEADS, FF_DIM, dt


if __name__ == '__main__':
    fit_model = False

    x_train, y_train, x_val, y_val, max_traj_steps, X, Y = get_data(data_num=2, max_traj_steps=MAX_TRAJ_STEPS, embed_dim=EMBED_DIM, dt=dt, debug=False)

    model = TrajectoryClassifier(max_traj_steps=max_traj_steps, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, ff_dim=FF_DIM)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    if fit_model:
        history = model.fit(x_train, y_train, 
                            batch_size=1, epochs=100, 
                            validation_data=(x_val, y_val)
                           )
        model.save_weights("weights/predict_class_real_data.h5")
    else:
        model.build((None, max_traj_steps, EMBED_DIM))
        model.load_weights("weights/predict_class_real_data.h5")

    print("Evaluation Result here: ")
    results = model.evaluate(x_val, y_val, verbose=2)
    for name, value in zip(model.metrics_names, results):
        print("%s: %.3f" % (name, value))

    
    # Achieved 0.833 accuaracy on the validation set!!!
    
    # exclude_indexes_list = [4, 6, 11, 12, 13, 14, 15, 16, 22, 23, 25, 31]
    # safe_data_index_range = (1, 21)
    # unsafe_data_index_range = (21, 35)
    print("Some test data: ")
    for i in range(X.shape[0]):
        print(f"True label: {Y[i]}, Predicted label: {model.predict(X[np.newaxis, i])}")


