from trajectory_model.data import get_data
from trajectory_model.model import TrajectoryClassifier
from trajectory_model.constants import MAX_TRAJ_STEPS, EMBED_DIM, NUM_HEADS, FF_DIM

# Train index:  [158  46 129  70  19 113  35 167  77 125  85  53  50  13 170  59 160  95
#  105  24 156  75 168  89  43 161  91  88 162 146 123   0  22 118  92 175
#   18  65  52  47  58  96  42 153  16 172  37  40 139  66 143  28 126  61
#  128  56 104 150  69  36 119 180  62  72   9  93  81  76 120  29  80 102
#    2  84 124 145 149  27  79  54  44 163   1  87  73  82 109 177  86 173
#  116  67  97 117 106  31 136   6 147  14 114  30  60  90 152 127   7 155
#  176 171  49 169  94  11 112  41 140 151  45  34  26  68  78 103 141 107
#   51  57  98 101  20  32 164 122  55  74 137 179]
# val index:  [  3   4   5   8  10  12  15  17  21  23  25  33  38  39  48  63  64  71
#   83  99 100 108 110 111 115 121 130 131 132 133 134 135 138 142 144 148
#  154 157 159 165 166 174 178 181 182 183]

if __name__ == '__main__':
    fit_model = True

    x_train, y_train, x_val, y_val, X, Y = get_data()
    model = TrajectoryClassifier(max_traj_steps=MAX_TRAJ_STEPS, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, ff_dim=FF_DIM)
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
        model.save_weights(f'weights/acc_{round(acc_val, 2)}_loss_{round(loss_val, 2)}_data_num_{training_data_num}_epochs_{epochs}.h5')

        eval_tr = model.evaluate(x_train, y_train, verbose=2)
        acc_tr, loss_tr = eval_tr[1],  eval_tr[0]
        print(f'Training: accuracy: {round(acc_tr, 2)}, loss: {round(loss_tr, 2)}')
        print(f'Validation: accuracy: {round(acc_val, 2)}, loss: {round(loss_val, 2)}')
        print(f'Number of training data: {training_data_num}, epochs: {epochs}')
        print(f'Saved model to disk.')
    else:
        model.build((None, MAX_TRAJ_STEPS, EMBED_DIM))
        model.load_weights("weights/predict_class_real_data_latest.h5")
        eval = model.evaluate(x_val, y_val, verbose=2)
        accuracy, loss = eval[1], eval[0]
        print(f'Loded model from disk. accuracy: {round(accuracy, 2)}, loss: {round(loss, 2)}.')
    
    # print("Testing: ")
    # for i in range(X.shape[0]//4):
    #     print(f"True label: {Y[i]}, Predicted label: {model.predict(X[np.newaxis, i])}")

