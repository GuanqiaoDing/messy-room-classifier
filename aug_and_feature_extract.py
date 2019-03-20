from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import Xception
import numpy as np

# load data
(x_train, y_train), (x_val, y_val) = np.load('data/room_dataset.npy')

# normalize data
channel_mean = np.mean(x_train, axis=(0, 1, 2))
channel_std = np.std(x_train, axis=(0, 1, 2))

# mean, std calculated here are also used in predicting test images: predict.py
# channel_mean = np.array([110.73151039, 122.90935242, 136.82249855])
# channel_std = np.array([69.39734207, 67.48444001, 66.66808662])

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')

for i in range(3):
    x_train[:, :, :, i] = (x_train[:, :, :, i] - channel_mean[i]) / channel_std[i]
    x_val[:, :, :, i] = (x_val[:, :, :, i] - channel_mean[i]) / channel_std[i]

# define data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

augmented_data = x_train.copy()
train_labels = y_train.copy()

# flow in advance, get augmented training data and corresponding labels
for i in range(19):
    for img, label in datagen.flow(x_train, y_train, batch_size=192):
        print(i)
        augmented_data = np.vstack((augmented_data, img))
        print(augmented_data.shape)
        train_labels = np.hstack((train_labels, label))
        print(train_labels.shape)
        break

# pre-trained model to extract features
base_model = Xception(include_top=False, weights='imagenet', pooling='avg')
train_features = base_model.predict(augmented_data)
np.savetxt('data/train_features.csv', train_features, fmt='%.5f', delimiter=',')
np.savetxt('data/train_labels.csv', train_labels, fmt='%1d', delimiter=',')

# save validation features and label
val_features = base_model.predict(x_val)
np.savetxt('data/val_features.csv', val_features, fmt='%.5f', delimiter=',')
np.savetxt('data/val_labels.csv', y_val, fmt='%1d', delimiter=',')
