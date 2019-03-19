import os
import cv2 as cv
from glob import glob

categories = ['clean', 'messy']
raw_dir = {'train': './raw/train', 'val': './raw/val'}
output_dir = './images'
extensions = ('*.jpg', '*.png')
img_size = 299  # match Xception input size


def resize(image):
    h, w, c = image.shape
    cropped = image
    if h < w:
        diff = (w - h) // 2
        cropped = image[:, diff: (diff + h), :]
    elif h > w:
        diff = (h - w) // 2
        cropped = image[diff: (diff + w), :, :]

    h, w, c = cropped.shape
    if h > img_size:    # shrink
        return cv.resize(cropped, (img_size, img_size), interpolation=cv.INTER_AREA)
    elif h < img_size:  # enlarge
        return cv.resize(cropped, (img_size, img_size), interpolation=cv.INTER_CUBIC)
    else:
        return cropped


if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for dataset, path in raw_dir.items():
    output_set_dir = os.path.join(output_dir, dataset)
    if not os.path.exists(output_set_dir):
        os.mkdir(output_set_dir)

    for cat in categories:
        output_cat_dir = os.path.join(output_set_dir, cat)
        if not os.path.exists(output_cat_dir):
            os.mkdir(output_cat_dir)

        input_dir = os.path.join(path, cat)
        filenames = list()
        for ext in extensions:
            filenames.extend(glob(os.path.join(input_dir, ext)))

        for i, file in enumerate(filenames):
            print('processing:', file)
            img = cv.imread(file)
            resized = resize(img)
            img_name = str(i) + '.png'
            filepath = os.path.join(output_cat_dir, img_name)
            cv.imwrite(filepath, resized)

