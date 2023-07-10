from trajectory_model.data import get_data
from trajectory_model.model import TrajectoryClassifier
from trajectory_model.constants import MAX_TRAJ_STEPS, EMBED_DIM, NUM_HEADS, FF_DIM, dt


if __name__ == '__main__':
    fit_model = True

    x_train, y_train, x_val, y_val, max_traj_steps, X, Y = get_data(data_num=2, max_traj_steps=MAX_TRAJ_STEPS, embed_dim=EMBED_DIM, dt=dt, debug=False)
    model = TrajectoryClassifier(max_traj_steps=max_traj_steps, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, ff_dim=FF_DIM)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    if fit_model:
        epochs = 350
        history = model.fit(x_train, y_train, 
                        batch_size=1, epochs=400, 
                        validation_data=(x_val, y_val)
                        )
        accuracy = model.evaluate(x_val, y_val, verbose=2)[1]
        training_data_num = x_train.shape[0]
        model.save_weights(f'weights/acc_{round(accuracy, 2)}_data_num_{training_data_num}.h5')
    else:
        model.build((None, max_traj_steps, EMBED_DIM))
        model.load_weights("weights/predict_class_real_data_latest.h5")
    
    results = model.evaluate(x_val, y_val, verbose=2)
    print("Evaluation:")
    for name, value in zip(model.metrics_names, results):
        print("%s: %.3f" % (name, value))
    
    # print("Testing: ")
    # for i in range(X.shape[0]//4):
    #     print(f"True label: {Y[i]}, Predicted label: {model.predict(X[np.newaxis, i])}")

