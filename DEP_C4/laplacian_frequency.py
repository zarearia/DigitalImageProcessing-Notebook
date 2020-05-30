import matplotlib.pyplot as plt
import numpy as np
import cv2
import utilities

# MARK: the image

img = cv2.imread('/Users/ariazare/Projects/Python/DEP_C4/Fig0458(a)(blurry_moon).tif')
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

img_padded = np.zeros((img.shape[0] * 2, img.shape[1] * 2), 'float32')
img_padded[:img.shape[0], :img.shape[1]] = img

img_padded = utilities.center_frequency(img_padded)

complex_img = [img_padded, np.zeros(img_padded.shape, 'float32')]
complex_img = cv2.merge(complex_img)
cv2.dft(complex_img, complex_img)

# cv2.imshow('test', img)
# cv2.waitKey(0)

# MARK: the filter


def creat_laplacian_filter_frequency(full_shape):
    laplacian_filter = np.zeros(full_shape, 'float32')
    filter_p, filter_q = full_shape[0], full_shape[1]
    half_p = filter_p // 2
    half_q = filter_q // 2
    for i in range(filter_p):
        for j in range(filter_q):
            laplacian_filter[i, j] = -4 * np.square(np.pi) * (np.square(i-half_p) + np.square(j-half_q))
            # laplacian_filter[i, j] = -4 * np.square(np.pi) * (np.square(i) + np.square(j))
    plt.imshow(laplacian_filter, 'Greys')
    plt.show()
    return laplacian_filter


laplacian_filter = creat_laplacian_filter_frequency(img_padded.shape)
laplacian_filter = [laplacian_filter, laplacian_filter]
laplacian_filter = cv2.merge(laplacian_filter)

filtered_image = laplacian_filter * complex_img
cv2.idft(filtered_image, filtered_image)
filtered_image = filtered_image[:img.shape[0], :img.shape[1], 0]
filtered_image = utilities.center_frequency(filtered_image)

plt.imshow(filtered_image, 'Greys')
plt.show()

cv2.imshow('img', img)
cv2.imshow('filtered', filtered_image)
cv2.waitKey(0)
























