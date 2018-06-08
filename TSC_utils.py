import cv2
import numpy as np


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