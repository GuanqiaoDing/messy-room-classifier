from keras import utils
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import Xception
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import numpy as np
import model

# categories = ['clean', 'messy']
epochs = 100
batch_size = 32
iterations = 6     # 192 / 32
lr = 5e-4
lr_decay = 5e-4

# load data
(x_train, y_train), (x_test, y_test) = np.load('dataset.npy')

# normalize data
channel_mean = np.mean(x_train, axis=(0, 1, 2))
channel_std = np.std(x_train, axis=(0, 1, 2))
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

for i in range(3):
    x_train[:, :, :, i] = (x_train[:, :, :, i] - channel_mean[i]) / channel_std[i]
    x_test[:, :, :, i] = (x_test[:, :, :, i] - channel_mean[i]) / channel_std[i]

# labels to categorical
# y_train = utils.to_categorical(y_train, len(categories))
# y_test = utils.to_categorical(y_test, len(categories))

# data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
datagen.fit(x_train)

# build a classifier from pre-trained model
base_model = Xception(include_top=False, weights='imagenet')
classifier = model.tune_model(base_model)
classifier.compile(optimizer=Adam(lr=lr, decay=lr_decay), loss='binary_crossentropy', metrics=['accuracy'])

# training
classifier.fit_generator(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=iterations,
    epochs=epochs,
    validation_data=(x_test, y_test),
    callbacks=[TensorBoard(log_dir='./log/03.18.19')],
    verbose=1
)
