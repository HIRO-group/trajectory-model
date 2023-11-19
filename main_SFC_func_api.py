import numpy as np

import tensorflow as tf
from trajectory_model.data import get_data
from trajectory_model.spill_free.model_func_api import get_SFC_model
from trajectory_model.helper.model_helper import SaveBestAccuracy
from trajectory_model.helper.helper import ctime_str
from trajectory_model.helper.helper import plot_loss_function

if __name__ == '__main__':
    fit_model = True
    X_train, Y_train, X_val, Y_val, X, Y = get_data(model_name='SFC', manual=False)

    X_train_traj = X_train[:, :, :7]
    X_train_prop = X_train[:, 0, 7:]
    X_val_traj = X_val[:, :, :7]
    X_val_prop = X_val[:, 0, 7:]
    
    model = get_SFC_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    
    if fit_model:
        epochs = 3
        batch_size = 32
        custom_cb = SaveBestAccuracy(file_address="spill_classifier_func_api", 
                                        min_val_acc=0.89,
                                        min_train_acc=0.89,
                                        model=model,
                                        X_train_traj=X_train_traj,
                                        X_train_prop=X_train_prop,
                                        Y_train=Y_train,
                                        )
        history = model.fit(
                        {"trajectory": X_train_traj,
                        "properties": X_train_prop,},
                        {"prediction": Y_train},
                        batch_size=batch_size,
                        epochs=epochs, 
                        callbacks=[custom_cb],
                        )

        eval_val = model.evaluate({"trajectory": X_val_traj, 
                                    "properties": X_val_prop,},
                                    {"prediction": Y_val}, verbose=2)

        acc_val, loss_val = eval_val[1], eval_val[0]
        training_data_num = X_train.shape[0]
        model.save_weights(f'weights/spill_classifier_func_api/{ctime_str()}_acc_{round(acc_val, 2)}_loss_{round(loss_val, 2)}_data_num_{training_data_num}_epochs_{epochs}.h5')

        eval_tr = model.evaluate( {"trajectory": X_train_traj,
                                    "properties": X_train_prop,},
                                    {"prediction": Y_train}, verbose=2)
        acc_tr, loss_tr = eval_tr[1],  eval_tr[0]


        plot = input("Plot? (y/n): ")
        if plot == 'y':
            plot_loss_function(history)

        print(f'Training: accuracy: {round(acc_tr, 2)}, loss: {round(loss_tr, 2)}')
        print(f'Validation: accuracy: {round(acc_val, 2)}, loss: {round(loss_val, 2)}')
        print(f'Number of training data: {training_data_num}, epochs: {epochs}')
        print(f'Saved model to disk.')
    else:
        model.load_weights("/home/ava/projects/trajectory-model/weights/spill_classifier_func_api/best/2023-10-31 20:11:14_epoch_169_train_acc_0.91.h5")
        eval = model.evaluate({"trajectory": X_val_traj, 
                                "properties": X_val_prop,},
                                {"prediction": Y_val}, verbose=2)
        loss, accuracy, precision, recall = eval[0], eval[1], eval[2], eval[3]

        print("Loss is: ", loss)
        print("Accuracy is: ", accuracy)
        print("Precision is: ", precision)
        print("Recall is: ", recall)

        prediction = model.predict({"trajectory": X[0, :, :7][None, :, :],
                                    "properties": X[0, 0, 7:][None, :],
                                    })[0][0]
        actual_value = Y[0]
        print("prediction: ", prediction, "actual value: ", actual_value)