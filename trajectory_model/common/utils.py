from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_address(args, acc, loss, epochs=None, prefix=None):
    epochs = epochs or args.epochs
    path_prefix = f"{args.save_addr_prefix}/{prefix}" if prefix else args.save_addr_prefix
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    name = f"{path_prefix}/{time}_a{round(acc, 2)}_l{round(loss, 2)}_e{epochs}.h5"
    return name


def plot_loss_function(history):
    dt = datetime.now()
    now = dt.strftime("%H:%M:%S")

    train_loss = history.history['loss']
    # val_loss = history.history['val_loss']
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, 'b', label='Training Loss')
    # plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'artifacts/plots/loss/loss_{now}.png')
    plt.show()
