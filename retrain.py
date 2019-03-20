from keras import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import time


epochs = 10
batch_size = 32
iterations = 120     # 192 * 20 / 32
weight_decay = 0.01

# load feature vectors and labels
x_train = np.genfromtxt('data/train_features.csv', dtype=np.float32, delimiter=',')
x_val = np.genfromtxt('data/val_features.csv', dtype=np.float32, delimiter=',')
y_train = np.genfromtxt('data/train_labels.csv', dtype=np.uint8)
y_val = np.genfromtxt('data/val_labels.csv', dtype=np.uint8)

# custom head: one hidden layer.
model = Sequential([
    Dense(10, activation='sigmoid', kernel_regularizer=regularizers.l2(weight_decay)),
    Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(weight_decay))
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# set up callback
cur_time = str(int(time.time()))
cbks = [
    TensorBoard(log_dir='./log/room_' + cur_time),
    ModelCheckpoint('./ckpt/' + cur_time + '_{epoch:02d}-{val_loss:.2f}.hdf5', save_best_only=True)
]

# training
history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    shuffle=False,      # already shuffled during augmentation
    validation_data=(x_val, y_val),
    callbacks=cbks,
    verbose=1
)

# save and plot result
model.save('./model/room_model_{}.h5'.format(cur_time))

train_error = [(1-acc)*100 for acc in history.history['acc']]
val_error = [(1-acc)*100 for acc in history.history['val_acc']]

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
plt.tight_layout(pad=3, w_pad=2)
fig.suptitle('Messy Room Classifier', fontsize=16, fontweight='bold')
ax1.set_xlabel('Epochs', fontsize=14)
ax1.set_ylabel('Error(%)', fontsize=14)
ax1.plot(train_error, label='Training Error')
ax1.plot(val_error, label='Validation Error')
ax1.legend()

ax2.set_xlabel('Epochs', fontsize=14)
ax2.set_ylabel('Loss', fontsize=14)
ax2.plot(history.history['loss'], label='Training Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.legend()

plt.savefig('./model/room_model_{}.png'.format(cur_time))

