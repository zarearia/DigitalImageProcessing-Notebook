import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('/Users/ariazare/Projects/Python/DEP_C4/Fig0424(a)(rectangle).tif')
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

img = cv2.pyrDown(img)
img = cv2.pyrDown(img)
img = cv2.pyrDown(img)
img = cv2.pyrDown(img)
img = cv2.pyrDown(img)

array = [np.array(img, 'float32'), np.zeros(img.shape, 'float32')]

complex = cv2.merge(array)


def my_2d_dft(input, output):
    for i in range(0, input.shape[0]):
        for j in range(0, input.shape[1]):
            sum = 0
            for k in range(0, input.shape[0]):
                for l in range(0, input.shape[1]):
                    e = np.exp(-1j * (np.pi * 2) * (i * k / input.shape[0] + j * l / input.shape[1]))
                    sum += e * input[k, l, 0]
            output[i, j, 0] = sum.real
            output[i, j, 1] = sum.imag
    return output


my_2d_dft_array = np.zeros(complex.shape, 'float32')
my_2d_dft_array = my_2d_dft(complex, my_2d_dft_array)

print(my_2d_dft_array)

plt.plot(my_2d_dft_array[0])
plt.show()
