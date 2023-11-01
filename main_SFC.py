import tensorflow as tf
from trajectory_model.data import get_data
from trajectory_model.spill_free.model import TrajectoryClassifier
from trajectory_model.spill_free.constants import MAX_TRAJ_STEPS, EMBED_DIM, NUM_HEADS, FF_DIM
from trajectory_model.helper.model_helper import SaveBestAccuracy
from trajectory_model.helper.helper import ctime_str

if __name__ == '__main__':
    fit_model = False
    X_train, Y_train, X_val, Y_val, X, Y = get_data(model_name='SFC', manual=False)
    model = TrajectoryClassifier(max_traj_steps=MAX_TRAJ_STEPS, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, ff_dim=FF_DIM)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    if fit_model:
        epochs = 200
        batch_size = 16
        custom_cb = SaveBestAccuracy(file_address="spill_classifier", 
                                        min_val_acc=0.87,
                                        min_train_acc=0.87)
        history = model.fit(X_train, Y_train, 
                        batch_size=batch_size, epochs=epochs, 
                        validation_data=(X_val, Y_val),
                        callbacks=[custom_cb],
                        )
        eval_val = model.evaluate(X_val, Y_val, verbose=2)
        acc_val, loss_val = eval_val[1], eval_val[0]
        training_data_num = X_train.shape[0]
        model.save_weights(f'weights/spill_classifier/{ctime_str()}_acc_{round(acc_val, 2)}_loss_{round(loss_val, 2)}_data_num_{training_data_num}_epochs_{epochs}.h5')

        eval_tr = model.evaluate(X_train, Y_train, verbose=2)
        acc_tr, loss_tr = eval_tr[1],  eval_tr[0]
        print(f'Training: accuracy: {round(acc_tr, 2)}, loss: {round(loss_tr, 2)}')
        print(f'Validation: accuracy: {round(acc_val, 2)}, loss: {round(loss_val, 2)}')
        print(f'Number of training data: {training_data_num}, epochs: {epochs}')
        print(f'Saved model to disk.')
    else:
        model.build((None, MAX_TRAJ_STEPS, EMBED_DIM))
        model.load_weights("/home/ava/projects/trajectory-model/weights/spill_classifier/best/2023-09-09 14:42:38_epoch_191_best_val_acc_0.93_train_acc_0.92.h5")
        eval = model.evaluate(X_val, Y_val, verbose=2)
        loss, accuracy, precision, recall = eval[0], eval[1], eval[2], eval[3]
        print("Loss is: ", loss)
        print("Accuracy is: ", accuracy)
        print("Precision is: ", precision)
        print("Recall is: ", recall)
