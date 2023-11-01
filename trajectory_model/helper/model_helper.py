import tensorflow as tf
from trajectory_model.helper.helper import ctime_str

class SaveBestAccuracy(tf.keras.callbacks.Callback):
    def __init__(self, file_address, model, X_train_traj, X_train_prop, 
                 Y_train, min_val_acc=0.87, min_train_acc=0.87):
        super().__init__()
        self.file_address = file_address
        self.min_val_acc = min_val_acc
        self.min_train_acc = min_train_acc

    def on_train_begin(self, logs=None):
        self.val_acc = [0.5]

    def on_epoch_end(self, epoch, logs=None):
        # val_loss,val_mae = model.evaluate(x_val,y_val)

        min_epoch = 0

        current_train_acc = logs.get("accuracy")
        
        # print("logs keys: ", logs.keys())
        # current_val_acc = logs.get("val_accuracy")
        # self.val_acc.append(logs.get("val_accuracy"))
        # print("logs: ", logs)
        # print("Current val accuracy: ", current_val_acc)
        # if current_val_acc >= max(self.val_acc) and \
            # current_val_acc >= self.min_val_acc and \
        if current_train_acc >= self.min_train_acc:
            min_epoch = epoch
            print(f'Found best accuracy. Saving entire model. Epoch: {min_epoch}')
            # print('Val accuracy: ', current_val_acc, ', Train accuracy: ', current_train_acc)
            self.model.save_weights(f'weights/{self.file_address}/best/{ctime_str()}_epoch_{min_epoch}_train_acc_{round(current_train_acc, 2)}.h5')