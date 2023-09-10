from trajectory_model.data import get_data
from trajectory_model.spill_free.model import TrajectoryClassifier
from trajectory_model.spill_free.constants import MAX_TRAJ_STEPS, EMBED_DIM, NUM_HEADS, FF_DIM
from trajectory_model.helper import SaveBestAccuracy, ctime_str

if __name__ == '__main__':
    fit_model = True
    x_train, y_train, x_val, y_val, X, Y = get_data()

    model = TrajectoryClassifier(max_traj_steps=MAX_TRAJ_STEPS, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, ff_dim=FF_DIM)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    if fit_model:
        epochs = 200
        custom_cb = SaveBestAccuracy("spill_classifier")
        history = model.fit(x_train, y_train, 
                        batch_size=16, epochs=epochs, 
                        validation_data=(x_val, y_val),
                        callbacks=[custom_cb],
                        )
        eval_val = model.evaluate(x_val, y_val, verbose=2)
        acc_val, loss_val = eval_val[1], eval_val[0]
        training_data_num = x_train.shape[0]
        model.save_weights(f'weights/spill_classifier/{ctime_str()}_acc_{round(acc_val, 2)}_loss_{round(loss_val, 2)}_data_num_{training_data_num}_epochs_{epochs}.h5')

        eval_tr = model.evaluate(x_train, y_train, verbose=2)
        acc_tr, loss_tr = eval_tr[1],  eval_tr[0]
        print(f'Training: accuracy: {round(acc_tr, 2)}, loss: {round(loss_tr, 2)}')
        print(f'Validation: accuracy: {round(acc_val, 2)}, loss: {round(loss_val, 2)}')
        print(f'Number of training data: {training_data_num}, epochs: {epochs}')
        print(f'Saved model to disk.')
    else:
        model.build((None, MAX_TRAJ_STEPS, EMBED_DIM))
        model.load_weights("weights/spill_free/predict_class_real_data_latest.h5")
        eval = model.evaluate(x_val, y_val, verbose=2)
        accuracy, loss = eval[1], eval[0]
        print(f'Loded model from disk. accuracy: {round(accuracy, 2)}, loss: {round(loss, 2)}.')
