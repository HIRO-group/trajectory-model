import tensorflow as tf

from trajectory_model.common.utils import generate_address

class SaveBestAccuracy(tf.keras.callbacks.Callback):
    def __init__(self, args,
                 model=None,
                 X_train_traj=None,
                 X_train_prop=None, 
                 Y_train=None,
                 min_val_acc=0.89,
                 min_train_acc=0.89):
        super().__init__()
        self.args = args
        self.min_val_acc = min_val_acc
        self.min_train_acc = min_train_acc


    def on_epoch_end(self, epoch, logs=None):
        current_train_acc = logs.get("accuracy")
        if current_train_acc >= self.min_train_acc:
            save_address = generate_address(self.args, current_train_acc, logs.get("loss"), epochs=epoch)
            self.model.save_weights(save_address)
