# from process_data_wp import process_data
from trajectory_model.data import get_position_wp_data
from trajectory_model.informed_sampler.model import PositionSampler
from trajectory_model.informed_sampler.constants import EMBED_DIM_POS, NUM_HEADS, FF_DIM, MAX_NUM_WAYPOINTS

# print(X.shape) # 220, 10, 8  (num_trajectories) (num_of_waypoints) (pos, orientation, cup)
# print(Y.shape) # 220, 7 (pos, orientation)


if __name__ == "__main__":
    fit_model = True
    x_train, y_train, x_val, y_val, X, Y = get_position_wp_data()
    print(X.shape)
    print(Y.shape)
    input("Haha")
    
    model = PositionSampler(max_num_waypoints=MAX_NUM_WAYPOINTS, waypoints_embed_dim=EMBED_DIM_POS, num_heads=NUM_HEADS, ff_dim=FF_DIM)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    if fit_model:
        epochs = 80
        history = model.fit(x_train, y_train, 
                        batch_size=1, epochs=epochs, 
                        validation_data=(x_val, y_val)
                        )
        eval_val = model.evaluate(x_val, y_val, verbose=2)
        acc_val, loss_val = eval_val[1], eval_val[0]
        training_data_num = x_train.shape[0]
        model.save_weights(f'weights/sampler/acc_{round(acc_val, 2)}_loss_{round(loss_val, 2)}_data_num_{training_data_num}_epochs_{epochs}.h5')

        eval_tr = model.evaluate(x_train, y_train, verbose=2)
        acc_tr, loss_tr = eval_tr[1],  eval_tr[0]
        print(f'Training: accuracy: {round(acc_tr, 2)}, loss: {round(loss_tr, 2)}')
        print(f'Validation: accuracy: {round(acc_val, 2)}, loss: {round(loss_val, 2)}')
        print(f'Number of training data: {training_data_num}, epochs: {epochs}')
        print(f'Saved model to disk.')
    else:
        model.build((None, MAX_NUM_WAYPOINTS, EMBED_DIM))
        model.load_weights("weights/sampler/predict_class_real_data_latest.h5")
        eval = model.evaluate(x_val, y_val, verbose=2)
        accuracy, loss = eval[1], eval[0]
        print(f'Loded model from disk. accuracy: {round(accuracy, 2)}, loss: {round(loss, 2)}.')
