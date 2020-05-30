import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('/Users/ariazare/Projects/python/DEP_C5/DIP3E_CH05_Original_Images/Fig0508(a)(circuit-board-pepper-prob-pt1).tif')
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def contraharmonic_mean_filter(src, q, kernel_size):
    dst = np.zeros(src.shape, 'float32')
    src_height = src.shape[0] - (kernel_size[0] // 2) - 1
    src_width = src.shape[1] - (kernel_size[1] // 2) - 1
    for i in range((kernel_size[0] // 2) + 1, src_height):
        for j in range((kernel_size[1] // 2) + 1, src_width):
            sum_1 = 0
            sum_2 = 0
            for k in range(-(kernel_size[0]//2), kernel_size[0]//2):
                for l in range(-(kernel_size[1]//2), kernel_size[1]//2):
                    sum_1 += np.power(src[i+k, j+l], q+1)
                    sum_2 += np.power(src[i+k, j+l], q)
            if sum_2 == 0:
                dst[i, j] = 0
            else:
                dst[i, j] = sum_1 // sum_2
    return dst


filtered_image = contraharmonic_mean_filter(img, 1.5, (5, 5))

plt.imshow(filtered_image, 'Greys_r')
plt.show()