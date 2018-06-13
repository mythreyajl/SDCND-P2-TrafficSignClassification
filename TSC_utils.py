import cv2
import numpy as np
import random

def deblur(image, k, a):
    blur = cv2.blur(image, (k, k))
    diff = image-blur
    image = image + a * diff
    maxi = np.max(image)
    image = image/maxi
    return image


def brightness_normalization(image, white):
    maxi = np.max(image)
    mini = np.min(image)
    alpha = white/(maxi-mini)
    beta = -mini*alpha
    return alpha*image + beta


def preprocess(image, k, a, w):
    image = brightness_normalization(image, w)
    return deblur(image, 3, 0.5)


def normalize_data(data):
    op = []
    for d in data:
        gray = cv2.cvtColor(d, cv2.COLOR_RGB2GRAY)                
        #gray = (gray-128)/128
        #gray = brightness_normalization(gray, 1.0)
        gray = np.expand_dims(gray, axis=2)
        op.append(gray)
    return op


def random_translator(X_in, max_delta):
    col, row, ch = X_in[0].shape
    X_mod = []
    for X in X_in:
        x = np.random.rand(col + 2 * max_delta, row + 2 * max_delta, ch)
        delta_x = max_delta + random.randint(-max_delta, max_delta)
        delta_y = max_delta + random.randint(-max_delta, max_delta)
        x[delta_x:delta_x + col, delta_y:delta_y + row, :] = X
        x = x[max_delta:max_delta + col, max_delta:max_delta + row, :]
        X_mod.append(x)
    return X_mod


def random_translation(X_in, y_in, max_delta):
    X_op = []
    y_op = []
    for X, y in zip(X_in, y_in):
        X_mod = random_translator(X, max_delta)
        X_op.append(X_mod)
        y_op.append(y)
    return X_op, y_op


def random_resizer(X_in, range):
    col, row, ch = X_in[0].shape
    X_mod = []
    for X in X_in:
        sx = random.uniform(range[0], range[1])
        sy = random.uniform(range[0], range[1])
        M = np.float32([[sx, 0, 0], [0, sy, 0]])
        x = cv2.warpAffine(X, M, (col, row))
        x = np.expand_dims(x, axis=2)
        X_mod.append(x)
    return X_mod


def random_resize(X_in, y_in, range):
    X_op = []
    y_op = []
    for X, y in zip(X_in, y_in):
        X_mod = random_resizer(X, range)
        X_op += X_mod
        y_op.append(y)
    return X_op, y_op


def random_rotator(X_in, range):
    col, row, ch = X_in[0].shape
    X_mod = []
    for X in X_in:
        rot = random.uniform(-range, range)
        M = cv2.getRotationMatrix2D((col/2, row/2), rot, 1)
        x = cv2.warpAffine(X, M, (col, row))
        x = np.expand_dims(x, axis=2)
        X_mod.append(x)
    return X_mod


def random_rotations(X_in, y_in, range):
    X_op = []
    y_op = []
    for X, y in zip(X_in, y_in):
        X_mod = random_rotator(X, range)
        X_op += X_mod
        y_op.append(y)
    return X_op, y_op
