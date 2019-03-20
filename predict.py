from keras.applications.xception import Xception
from keras.models import load_model
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os

# pre-process test images
# import preprocessing
# test_dir = './images/test'
# filenames = glob(os.path.join(test_dir, '*.jpg'))
#
# for i, file in enumerate(filenames):
#     print('processing:', file)
#     img = cv.imread(file)
#     resized = preprocessing.resize(img)
#     img_name = str(i) + '.png'
#     filepath = os.path.join(test_dir, img_name)
#     cv.imwrite(filepath, resized)


def load_test_images(file_list):
    test_set = list()
    test_set_rgb = list()
    for file in file_list:
        print(file)
        img = cv.imread(file)
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        test_set.append(img)
        test_set_rgb.append(img_rgb)

    return np.asarray(test_set), np.asarray(test_set_rgb)


# load test images
test_dir = './images/test'
filenames = glob(os.path.join(test_dir, '*.png'))
images, images_rgb = load_test_images(filenames)

# calculate from the training set
channel_mean = np.array([110.73151039, 122.90935242, 136.82249855])
channel_std = np.array([69.39734207, 67.48444001, 66.66808662])

# normalize images
images = images.astype('float32')
for j in range(3):
    images[:, :, :, j] = (images[:, :, :, j] - channel_mean[j]) / channel_std[j]

# make predictions
base_model = Xception(include_top=False, weights='imagenet', pooling='avg')
room_model = load_model('./model/room_model_1552970840.h5')
features = base_model.predict(images)
predictions = room_model.predict(features)

# plot results
fig = plt.figure()
fig.suptitle('Predictions on Test Images', size=15, weight='bold')
fig.subplots_adjust(hspace=0.3, wspace=0.2)
for i in range(10):
    ax = fig.add_subplot(3, 4, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(images_rgb[i], aspect='auto')
    result = 'Messy Prob: {:.2f}'.format(predictions[i][0])
    ax.set_xlabel(result, color='g', size=10, weight='bold', horizontalalignment='center')

plt.show()
