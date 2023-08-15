import numpy as np
from trajectory_model.data import get_orientation_wp_data
from trajectory_model.informed_sampler.model import OrientationSampler
from trajectory_model.informed_sampler.constants import EMBED_DIM_ORIENTATION_X, EMBED_DIM_ORIENTATION_Y, MAX_NUM_WAYPOINTS
from trajectory_model.helper import SaveBestAccuracy, plot_loss_function, plot_prediction_vs_real, plot_xyz, ctime_str


if __name__ == "__main__":
    fit_model = False
    x_train, y_train, x_val, y_val, X, Y = get_orientation_wp_data(manual=True)

    model = OrientationSampler(max_num_waypoints=MAX_NUM_WAYPOINTS,
                               embed_X=EMBED_DIM_ORIENTATION_X,
                               embed_Y=EMBED_DIM_ORIENTATION_Y)

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    if fit_model:
        epochs = 5000
        custom_cb = SaveBestAccuracy()
        history = model.fit(x_train, y_train,
                            batch_size=32, epochs=epochs,
                            validation_data=(x_val, y_val),
                            callbacks=[custom_cb],
                            )
        eval_val = model.evaluate(x_val, y_val, verbose=2)
        acc_val, loss_val = eval_val[1], eval_val[0]
        training_data_num = x_train.shape[0]

        save = input("Save weights? (y/n): ")
        if save == 'y':
            model.save_weights(
                f'weights/orientation_sampler/{ctime_str()}_acc_{round(acc_val, 2)}_loss_{round(loss_val, 2)}_data_num_{training_data_num}_epochs_{epochs}.h5')
            print(f'Saved model to disk.')

        plot = input("Plot? (y/n): ")
        if plot == 'y':
            plot_loss_function(history)

        print("Best epoch: ", history.history['val_accuracy'].index(max(history.history['val_accuracy'])))

        eval_tr = model.evaluate(x_train, y_train, verbose=2)
        acc_tr, loss_tr = eval_tr[1],  eval_tr[0]

        print(f'Training: accuracy: {round(acc_tr, 2)}, loss: {round(loss_tr, 2)}')
        print(f'Validation: accuracy: {round(acc_val, 2)}, loss: {round(loss_val, 2)}')
        print(f'Number of training data: {training_data_num}, epochs: {epochs}')

    else:
        model.build((None, MAX_NUM_WAYPOINTS, EMBED_DIM_ORIENTATION_X))
        model.load_weights(
            "weights/orientation_sampler/acc_0.85_loss_0.01_data_num_165_epochs_400.h5")
        eval = model.evaluate(x_val, y_val, verbose=2)
        accuracy, loss = eval[1], eval[0]

        eval_tr = model.evaluate(x_train, y_train, verbose=2)
        accuracy_tr, loss_tr = eval_tr[1], eval_tr[0]
        print("Loaded model from disk.")
        print(f'Validation accuracy: {round(accuracy, 2)}, loss: {round(loss, 2)}.')
        print(f'Training: accuracy: {round(accuracy_tr, 2)}, loss: {round(loss_tr, 2)}')

        # test_index = 1
        # x_test, y_test = x_val[test_index], y_val[test_index]
        # predicted_xyz = model.predict(x_test[np.newaxis, :])[0]
        # print("Predicted xyz: ", predicted_xyz)
        # print("True xyz: ", y_test)
        # plot_prediction_vs_real(predicted=predicted_xyz, real=y_test)
