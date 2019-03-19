from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import Xception
import numpy as np

# load data
(x_train, y_train), (x_test, y_test) = np.load('data/dataset.npy')

# normalize data
channel_mean = np.mean(x_train, axis=(0, 1, 2))
channel_std = np.std(x_train, axis=(0, 1, 2))
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

for i in range(3):
    x_train[:, :, :, i] = (x_train[:, :, :, i] - channel_mean[i]) / channel_std[i]
    x_test[:, :, :, i] = (x_test[:, :, :, i] - channel_mean[i]) / channel_std[i]

# data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

augmented_data = x_train.copy()
augmented_label = y_train.copy()

for i in range(19):
    for img, label in datagen.flow(x_train, y_train, batch_size=192):
        print(i)
        augmented_data = np.vstack((augmented_data, img))
        print(augmented_data.shape)
        augmented_label = np.hstack((augmented_label, label))
        print(augmented_label.shape)
        break

# pre-trained model to extract features
base_model = Xception(include_top=False, weights='imagenet', pooling='avg')
augmented_features = base_model.predict(augmented_data)
np.savetxt('data/augmented_features.csv', augmented_features, fmt='%.5f', delimiter=',')
np.savetxt('data/augmented_label.csv', augmented_label, fmt='%1d', delimiter=',')

# save test features and label
test_features = base_model.predict(x_test)
np.savetxt('data/test_features.csv', test_features, fmt='%.5f', delimiter=',')
np.savetxt('data/test_label.csv', y_test, fmt='%1d', delimiter=',')
