import numpy as np
import cv2
import matplotlib.pyplot as plt
import utilities
from mpl_toolkits.mplot3d import axes3d

# MARK: image
img = cv2.imread('/Users/ariazare/Projects/Python/DEP_C4/Fig0438(a)(bld_600by600).tif')
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_padded = np.zeros((602, 602), 'float32')
img_padded[:-2, 0:-2] = img

img_centered = utilities.center_frequency(img_padded)
complex_img = [np.array(img_centered, 'float32'), np.zeros(img_centered.shape, 'float32')]
complex_img = cv2.merge(complex_img)
cv2.dft(complex_img, complex_img)

# img_mag = utilities.get_magnitude(complex_img)
# img_mag = cv2.log(img_mag,img_mag)
# img_phase = utilities.get_phase_angel(complex_img)

# plt.imshow(img_mag, 'Greys')
# plt.show()

# MARK: sobel filter kernal
sobel_filter = np.array([[0, 0, 0, 0],
                         [0, -1, 0, 1],
                         [0, -2, 0, 2],
                         [0, -1, 0, 1]], 'float32')

sobel_filter_sp = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], 'float32')

sobel_filter_zeros = np.zeros((602, 602), 'float32')
sobel_filter_zeros[299:303, 299:303] = sobel_filter
# sobel_filter_zeros[299:302, 299:302] = sobel_filter
sobel_filter = sobel_filter_zeros

sobel_filter = utilities.center_frequency(sobel_filter)
sobel_filter_transformed = [sobel_filter, np.zeros(sobel_filter.shape, 'float32')]
sobel_filter_transformed = cv2.merge(sobel_filter_transformed)
cv2.dft(sobel_filter_transformed, sobel_filter_transformed)
sobel_filter_transformed[:, :, 0] = 0
sobel_filter_transformed[:, :] = utilities.center_frequency(sobel_filter_transformed[:, :])

# plt.plot(sobel_filter_transformed[:, :, 1])
# plt.show()

# plt.imshow(sobel_filter_transformed[:, :, 1], 'Greys')
# plt.show()

# MARK: filtering in frequency domain
# FIXME: Sraenge Probelm Here
filtered_img = np.copy(complex_img)
filtered_img = np.multiply(sobel_filter_transformed, complex_img)
filtered_img = cv2.idft(filtered_img, filtered_img)
filtered_img = utilities.center_frequency(filtered_img[:, :])

plt.imshow(filtered_img[:, :, 1], 'Greys')
plt.show()

# MARK: filtering in spatial domain
filtered_img_sp = np.zeros((600, 600), 'float32')
filtered_img_sp = cv2.filter2D(img, cv2.CV_32F, sobel_filter_sp)

# plt.imshow(filtered_img_sp, 'Greys')
# plt.show()






# FIXME: 3d ploting
# test 3d plot view

# figure = plt.figure()
# ax = figure.add_subplot(111, projection='3d')
# X, Y, Z = axes3d.get_test_data(0.05)
# ax.plot_wireframe(X,Y,Z, rstride=10, cstride=10)
# plt.show()

# test 2 3d plot view

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# test_data = sobel_filter_transformed[:, :, 1]
# # print(sobel_filter_transformed)
# x, y = np.mgrid[0:602], np.mgrid[0:602]
# ax.plot_trisurf(x, y, test_data, linewidth=0.2, antialiased=True)

# test 3 3d plot view

# x, y = np.mgrid[0:602], np.mgrid[0:602]
# test_data = sobel_filter_transformed[:, :, 1]
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(x, y, test_data, rstride=10, cstride=10, linewidth=0.2)
# plt.show()
















