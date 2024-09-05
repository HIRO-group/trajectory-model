from trajectory_model.SFC import get_SFC_model, SaveBestAccuracy
from trajectory_model.process_data import get_X_and_Y, get_train_and_val_sets
from trajectory_model.common import get_arguments, generate_address, plot_loss_function


def train_model(args, model, X, Y): 
    X_train_traj, X_train_prop, Y_train, X_val_traj, X_val_prop, Y_val = get_train_and_val_sets(X, Y)
    custom_cb = SaveBestAccuracy(args=args,
                                 model=model,
                                 X_train_traj=X_train_traj,
                                 X_train_prop=X_train_prop,
                                 Y_train=Y_train,)

    history = model.fit({"trajectory": X_train_traj, "properties": X_train_prop,}, {"prediction": Y_train},
                        batch_size=args.batch_size,
                        epochs=args.epochs, 
                        callbacks=[custom_cb])

    val_loss_acc = model.evaluate({"trajectory": X_val_traj, "properties": X_val_prop,}, {"prediction": Y_val}, verbose=2)

    save_address = generate_address(args=args, acc_val=val_loss_acc[1], loss_val=val_loss_acc[0])
    model.save_weights(save_address)

    if args.plot_loss_function:
        plot_loss_function(history)


    tr_loss_acc = model.evaluate({"trajectory": X_train_traj, "properties": X_train_prop,}, {"prediction": Y_train}, verbose=2)
    print(f'Training: accuracy: {round(tr_loss_acc[1], 2)}, loss: {round(tr_loss_acc[0], 2)}')
    print(f'Validation: accuracy: {round(val_loss_acc[1], 2)}, loss: {round(val_loss_acc[0], 2)}')
    print(f'Saved model to disk.')


def load_model(args, model, X, Y):
    _, _, _, X_val_traj, X_val_prop, Y_val = get_train_and_val_sets(X, Y)
    model.load_weights(args.load_weight_addr)
    eval = model.evaluate({"trajectory": X_val_traj, "properties": X_val_prop,},{"prediction": Y_val}, verbose=2)
    loss, accuracy = eval[0], eval[1]
    print("Loss is: ", loss)
    print("Accuracy is: ", accuracy)

    prediction = model.predict({"trajectory": X[0, :, :7][None, :, :], "properties": X[0, 0, 7:][None, :],})[0][0]
    actual_value = Y[0]
    print("prediction: ", prediction, "actual value: ", actual_value)


if __name__ == '__main__':
    args = get_arguments()
    X, Y = get_X_and_Y()
    model = get_SFC_model()
    
    if args.fit_model:
        train_model(args, model, X, Y)
    else:
        load_model(args, model, X, Y)