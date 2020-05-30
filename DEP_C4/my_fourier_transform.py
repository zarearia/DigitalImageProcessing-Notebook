import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

# Fourier Spectrum and Phase Angle
#########################################################################

# j = np.complex(0, 1)
# e = np.complex(np.e, 0)
img = cv2.imread('/Users/ariazare/Projects/Python/DEP_C4/Fig0424(a)(rectangle).tif')
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

cv2.imshow('Image', img)

img = cv2.pyrDown(img)
img = cv2.pyrDown(img)
img = cv2.pyrDown(img)
img = cv2.pyrDown(img)
print(img.shape)

# Expanding the image to an optimal size

rows, cols = img.shape
m = cv2.getOptimalDFTSize( rows )
n = cols = cv2.getOptimalDFTSize( cols )
padded_array = cv2.copyMakeBorder(img, 0, m - rows, 0, n - cols, cv2.BORDER_CONSTANT, value=[0, 0, 0])


# Make place for both the complex and the real values

planes = [np.float32(padded_array), np.zeros(padded_array.shape, np.float32)]
complex = cv2.merge(planes)         # Add to the expanded another plane with zeros

# Make the Discrete Fourier Transform

# complex2 = complex
#
# dft = cv2.dft(complex, complex)
# print(dft[0, 0:5])


def my_fourier_transform(src, output):
    for i in range(0, src.shape[0] - 1):
        for j in range(0, src.shape[1] - 1):
            sum = 0
            for m in range(0, src.shape[0] - 1):
                for n in range(0, src.shape[1] - 1):
                    sum += src[i, j] * np.exp(-2j * 2 * np.pi * (i * m / src.shape[0]) + j * n / src.shape[1])
            print(sum)
            output[i, j] = sum

            # output[i, j] =\
            #     src[i, j] * np.power(e, -2j * np.pi *
            #                          (np.divide(np.multiply(output[i], src[i]), src.shape[0]) +
            #                           np.divide(np.multiply(output[:, j], src[:, j]), src.shape[1])))

    return output


def creat_complex(array):
    complex_array = np.zeros(array.shape[:-1], 'complex')
    for i in range(0, array.shape[0]):
        for j in range(0, array.shape[1]):
            complex_array[i, j] = np.complex(array[i, j, 0], array[i, j, 1])
    return complex_array


complex_array = creat_complex(complex)


test = my_fourier_transform(complex, complex)
print(test)

print('the end')

#########################################################################


# cv2.waitKey(0)
# cv2.destroyAllWindows()

# [[ 1681980.           0.   ]
#  [-1678863.1     -15452.496]
#  [ 1669533.2      30735.836]
#  [-1654055.      -45682.81 ]
#  [ 1632533.8      60129.793]]