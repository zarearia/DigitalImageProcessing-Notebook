import numpy as np
import matplotlib.pyplot as plt
import cv2
import utilities


# TODO: The Image

img = cv2.imread('/Users/ariazare/Projects/Python/DEP_C4/Fig0441(a)(characters_test_pattern).tif')
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

img_padded = np.zeros((img.shape[0] * 2, img.shape[1] * 2), 'float32')
img_padded[:img.shape[0], :img.shape[1]] = img

img_padded = utilities.center_frequency(img_padded)

complex_img = [img_padded, np.zeros(img_padded.shape, 'float32')]
complex_img = cv2.merge(complex_img)
cv2.dft(complex_img, complex_img)


# MARK: The Filter

def creat_butterworth_filter(full_shape, radius, power):
    butterworth_filter = np.zeros(full_shape, 'float32')
    filter_p, filter_q = butterworth_filter.shape[0], butterworth_filter.shape[1]
    half_p = filter_p // 2
    half_q = filter_q // 2
    for i in range(filter_p):
        for j in range(filter_q):
            distance_from_origin = np.sqrt(np.square(i - half_p) + np.square(j - half_q))
            butterworth_filter[i, j] = np.divide(1, 1+(np.power(np.divide(distance_from_origin, radius), 2*power)))
    return butterworth_filter


























