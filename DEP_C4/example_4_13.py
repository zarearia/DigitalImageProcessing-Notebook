import numpy as np
import matplotlib.pyplot as plt
import cv2

np.set_printoptions(threshold=np.inf)


# Downloaded from Stackoverflow
def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def rearrange_frequency_spectrum(frequency_spectrum, shape):
    half_shape = shape // 2
    back_up_quarter = np.copy(frequency_spectrum[half_shape:shape, half_shape:shape])
    frequency_spectrum[half_shape:shape, half_shape:shape] = frequency_spectrum[0:half_shape, 0:half_shape]
    frequency_spectrum[0:half_shape, 0:half_shape] = back_up_quarter
    back_up_quarter2 = np.copy(frequency_spectrum[0:half_shape, half_shape:shape])
    frequency_spectrum[0:half_shape, half_shape:shape] = frequency_spectrum[half_shape:shape, 0:half_shape]
    frequency_spectrum[half_shape:shape, 0:half_shape] = back_up_quarter2
    return frequency_spectrum


img = cv2.imread('/Users/ariazare/Projects/Python/DEP_C4/Fig0424(a)(rectangle).tif')
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# img = rotateImage(img, -45)

raw_complex_array = [np.array(img, 'float32'), np.zeros(img.shape, 'float32')]
complex = cv2.merge(raw_complex_array)

cv2.dft(complex, complex)

complex_real = complex[0:1024, 0:1024, 0]
complex_imag = complex[0:1024, 0:1024, 1]

frequency_spectrum = np.sqrt(np.square(complex_real) + np.square(complex_imag))

frequency_spectrum = rearrange_frequency_spectrum(frequency_spectrum, 1024)

frequency_spectrum = np.log(1 + frequency_spectrum)

frequency_spectrum = cv2.normalize(frequency_spectrum, frequency_spectrum, 255, 0)

cv2.imshow('test2', frequency_spectrum)


phase_angel = np.arctan2(complex_real, complex_imag)

cv2.imshow('phase_angel', phase_angel)
cv2.waitKey(0)














