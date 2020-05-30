import numpy as np
import matplotlib.pyplot as plt
import cv2
import utilities

img = cv2.imread('/Users/ariazare/Projects/Python/DEP_C4/Fig0429(a)(blown_ic).tif')
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img[i, j] *= np.power(-1, i+j)

# img[img < 0] = 0

plt.imshow(img)
plt.show()
cv2.imshow('test0', img)

(complex_img, complex_img_real, complex_img_imag) = utilities.get_fourier_transform(img)
complex_img_mag = utilities.get_magnitude(complex_img)
complex_img_phase = utilities.get_phase_angel(complex_img)

complex_img_mag = utilities.rearrange_frequency_spectrum(complex_img_mag)

cv2.log(complex_img_mag, complex_img_mag)
plt.imshow(complex_img_mag, 'Greys')
plt.show()
complex_img_mag = cv2.normalize(complex_img_mag, complex_img_mag, 255, 0)
cv2.imshow('test', complex_img_mag)
cv2.waitKey(0)

# print(complex_img_mag.shape)
#
# complex_img[0:5, 0:5] = [0, 0]
# complex_img[672:678, 0:5] = [0, 0]
# complex_img[0:5, 900:906] = [0, 0]
# complex_img[672:678, 900:906] = [0, 0]
#
# complex_img_mag = utilities.get_magnitude(complex_img)
# complex_img_mag = utilities.rearrange_frequency_spectrum(complex_img_mag)
# cv2.log(complex_img_mag, complex_img_mag)
#
# plt.imshow(complex_img_mag)
# plt.show()
#
#
# cv2.idft(complex_img, complex_img)
# img_rebuild = complex_img[:, :, 0]
# img_rebuild_2 = np.copy(img_rebuild)
# # cv2.log(img_rebuild, img_rebuild_2)
# img_rebuild_2[img_rebuild_2 < 0] = 0
# cv2.normalize(img_rebuild, img_rebuild_2, 255, 0)
#
# plt.imshow(img_rebuild_2, 'Greys')
# plt.show()
#
# cv2.imshow('2', img_rebuild_2)
# cv2.waitKey(0)















