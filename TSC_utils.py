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
        gray = (gray-128)/128
        gray = brightness_normalization(gray, 1.0)
        gray = np.expand_dims(gray, axis=2)
        op.append(gray)
    return op


def random_translation(X_in, y_in, max_delta):
    X_op = []
    y_op = []
    col, row, ch = X_in[0].shape
    for X, y in zip(X_in, y_in):
        X_mod = np.random.rand(col+2*max_delta, row+2*max_delta, ch)
        delta_x = max_delta + random.randint(-max_delta, max_delta)
        delta_y = max_delta + random.randint(-max_delta, max_delta)
        X_mod[delta_x:delta_x+col, delta_y:delta_y+row, :] = X
        X_mod = X_mod[max_delta:max_delta+col, max_delta:max_delta+row, :]
        X_op.append(X_mod)
        y_op.append(y)
    return X_op, y_op


def random_resize(X_in, y_in, range):
    X_op = []
    y_op = []
    col, row, ch = X_in[0].shape
    for X, y in zip(X_in, y_in):
        sx = random.uniform(range[0], range[1])
        sy = random.uniform(range[0], range[1])
        M = np.float32([[sx, 0, 0], [0, sy, 0]])
        X_mod = cv2.warpAffine(X, M, (col, row))
        X_mod = np.expand_dims(X_mod, axis=2)
        X_op.append(X_mod)
        y_op.append(y)
    return X_op, y_op


def random_rotations(X_in, y_in, range):
    X_op = []
    y_op = []
    col, row, ch = X_in[0].shape
    for X, y in zip(X_in, y_in):
        rot = random.uniform(-range, range)
        M = cv2.getRotationMatrix2D((col/2, row/2), rot, 1)
        X_mod = cv2.warpAffine(X, M, (col, row))
        X_mod = np.expand_dims(X_mod, axis=2)
        X_op.append(X_mod)
        y_op.append(y)
    return X_op, y_op
