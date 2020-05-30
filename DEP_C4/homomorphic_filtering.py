import numpy as np
import matplotlib.pyplot as plt
import cv2
import utilities


# MARK: the image

img = cv2.imread('/Users/ariazare/Projects/Python/DEP_C4/Fig0462(a)(PET_image).tif')
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

img_padded = np.zeros((img.shape[0] * 2, img.shape[1] * 2), 'float32')
img_padded[:img.shape[0], :img.shape[1]] = img

img_padded = utilities.center_frequency(img_padded)

complex_img = [img_padded, np.zeros(img_padded.shape, 'float32')]
complex_img = cv2.merge(complex_img)
cv2.dft(complex_img, complex_img)


# MARK: the filter


def creat_homomorphic_filter(full_shape, yh, yl, c, radius):
    homomorphic_filter = np.zeros(full_shape, 'float32')
    filter_p, filter_q = homomorphic_filter.shape[0], homomorphic_filter.shape[1]
    half_p = filter_p // 2
    half_q = filter_q // 2
    for i in range(filter_p):
        for j in range(filter_q):
            homomorphic_filter[i, j] =\
                (yh - yl) * (1 - np.exp(-c * (np.square(i - half_p) + np.square(j - half_q)) / np.square(radius))) + yl
    return homomorphic_filter


# homomorphic_filter = creat_homomorphic_filter(img_padded.shape, 3.0, 0.4, 5, 20)
homomorphic_filter = creat_homomorphic_filter(img_padded.shape, 1.0, 0, 5, 20)
plt.imshow(homomorphic_filter, 'Greys_r')
plt.show()
plt.plot(homomorphic_filter)
plt.show()
homomorphic_filter = [homomorphic_filter, homomorphic_filter]
homomorphic_filter = cv2.merge(homomorphic_filter)


# MARK: filtering

filtered_image = complex_img * homomorphic_filter
cv2.idft(filtered_image, filtered_image)
filtered_image = filtered_image[:img.shape[0], :img.shape[1], 0]
filtered_image = utilities.center_frequency(filtered_image)
plt.imshow(filtered_image, 'Greys_r')
plt.show()
plt.imshow(img, 'Greys_r')
plt.show()



























