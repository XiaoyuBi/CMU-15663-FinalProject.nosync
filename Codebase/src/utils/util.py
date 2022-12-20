import numpy as np


def image_pixel_similarity(img1, img2, tol = 1e-5):
    assert img1.shape == img2.shape

    mask = (np.abs(img1 - img2) < tol)
    return mask.sum() / np.prod(img1.shape)


def PSNR(origin, test):
    
    MAX = np.max(origin)

    N = np.product(test.shape)
    MSE = np.sum((test - origin) ** 2) / N

    res = 20 * np.log10(MAX / np.sqrt(MSE))

    return res


def pad_to_shape(a, target_shape):

    x_target, y_target = target_shape
    x, y = a.shape
    x_pad = x_target - x
    y_pad = y_target - y

    return np.pad(a, 
    ((x_pad//2, x_pad//2 + x_pad%2), (y_pad//2, y_pad//2 + y_pad%2)))