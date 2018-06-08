# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = './traffic-signs-data/train.p'
validation_file = './traffic-signs-data/valid.p'
testing_file = './traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

############################################################################################################################################

import numpy as np

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Num as np

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

############################################################################################################################################

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import TSC_utils as utils
# Visualizations will be shown in the notebook.

# 1. Counts per class
"""
fig1 = plt.figure()
plt.subplot(3,1,1)
plt.hist(y_train, bins=n_classes, color='green')
plt.title("Distribution of training")
plt.show()
plt.subplot(3,1,2)
plt.hist(y_valid, bins=n_classes, color='green')
plt.title("Distribution of validation")
plt.show()
plt.subplot(3,1,3)
plt.hist(y_test, bins=n_classes, color='green')
plt.title("Distribution of testing")
plt.show()
"""

# 2. Per class mean image from training set
import string
import cv2

image_classes = {}
with open('./signnames.csv') as file:
    lines = file.readlines()
    for line in lines:
        mapping = line.split(',')
        if mapping[0] == 'ClassId':
            continue
        image_classes[int(mapping[0])] = {'name': mapping[1],
                                          'mean_img': 0.0 * np.ndarray(shape=image_shape, dtype=float),
                                          'images': []}

for x, y in zip(X_train, y_train):
    image_classes[y]['images'].append(x)

for y in np.unique(y_train):
    image_classes[y]['count'] = len(image_classes[y]['images'])
    img_array = np.asarray(image_classes[y]['images'])
    mean = np.mean(img_array, 0)
    image_classes[y]['mean_img'] = np.divide(mean.astype(float), 255.0)


"""
plt.figure(2, figsize=(7, 7))
img = 0.0 * np.ndarray(shape=(32 * 7, 32 * 7, 3), dtype=float)
for key, value in image_classes.items():
    plt.axis('off')
    quo = int(key / 7)
    rem = int(key % 7)
    img[quo * 32:(quo + 1) * 32, rem * 32:(rem + 1) * 32, :] = utils.preprocess(value['mean_img'], 3, 0.5)

plt.imshow(img)
plt.show()
"""
# 3. Animation of images per class
from matplotlib import animation
import random

fig = plt.figure(3, figsize=(5, 5))

img = np.ndarray(shape=(32*7, 32*7, 3), dtype=np.uint8)
im = plt.imshow(img)
plt.axis("off")
"""
def random_img(*args):
    for key in np.unique(y_train):
        label_dict = image_classes[key]
        size = label_dict['count']
        i = random.randint(0, size-1)
        rand_im = label_dict['images'][i]
        quo = int(key / 7)
        rem = int(key % 7)
        img[quo*32:(quo+1)*32, rem*32:(rem+1)*32, :] = rand_im
    im = plt.imshow(img)
    return im,

anim = animation.FuncAnimation(fig, random_img, repeat=True)
plt.show()
"""
############################################################################################################################################
X_train = utils.normalize_data(X_train)
X_valid = utils.normalize_data(X_valid)
X_test  = utils.normalize_data(X_test)



