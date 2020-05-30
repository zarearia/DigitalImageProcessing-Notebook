import numpy as np
import matplotlib.pyplot as plt
import cv2
import utilities
np.set_printoptions(threshold=np.inf)

# MARK: the image

img = cv2.imread('/Users/ariazare/Projects/Python/DEP_C4/Fig0441(a)(characters_test_pattern).tif')
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

img_padded = np.zeros((img.shape[0] * 2, img.shape[1] * 2), 'float32')
img_padded[:img.shape[0], :img.shape[1]] = img

img_padded = utilities.center_frequency(img_padded)

complex_img = [img_padded, np.zeros(img_padded.shape, 'float32')]
complex_img = cv2.merge(complex_img)
cv2.dft(complex_img, complex_img)


# MARK: the filter

# also exist in utilities
def creat_gaussian_lowpass_filter_frequency(full_shape, circle_radius):
    gaussian_filter = np.zeros(full_shape, 'float32')
    filter_p, filter_q = gaussian_filter.shape[0], gaussian_filter.shape[1]
    half_p = filter_p//2
    half_q = filter_q//2
    for i in range(filter_p):
        for j in range(filter_q):
            gaussian_filter[i, j]=\
                np.exp(-(np.square(i - half_p) + np.square(j - half_q)) / (2 * np.square(circle_radius)))
    return gaussian_filter


gaussian_filter = creat_gaussian_lowpass_filter_frequency(img_padded.shape, 10)
gaussian_filter = [gaussian_filter, gaussian_filter]
gaussian_filter = cv2.merge(gaussian_filter)


# MARK: filtering

filtered_image = complex_img * gaussian_filter
cv2.idft(filtered_image, filtered_image)
filtered_image = filtered_image[:img.shape[0], :img.shape[1], 0]
filtered_image = utilities.center_frequency(filtered_image)
plt.imshow(filtered_image, 'Greys_r')
plt.show()





















