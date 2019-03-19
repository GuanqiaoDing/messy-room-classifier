import os
from glob import glob
import cv2 as cv
import numpy as np

categories = ['clean', 'messy']
data_dir = {'train': './images/train', 'val': './images/val'}


def load_data():
    """96 images per class in training set, 10 images per class in validation set"""

    x_train = list()
    x_test = list()
    y_train = list()
    y_test = list()

    for dataset, path in data_dir.items():
        for i, cat in enumerate(categories):
            cur_dir = os.path.join(path, cat)
            filenames = glob(os.path.join(cur_dir, '*.png'))
            for file in filenames:
                img = cv.imread(file)
                if dataset == 'train':
                    x_train.append(img)
                    y_train.append(i)
                else:
                    x_test.append(img)
                    y_test.append(i)

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    return (x_train, y_train), (x_test, y_test)


# save the dataset
room_dataset = load_data()
np.save('dataset.npy', room_dataset)
