import numpy as np
import matplotlib.pyplot as plt
import cv2


array = np.zeros((1000), 'float32')
array[300:700] = 1
cv2_dft_array = np.zeros((1000), 'float32')


cv2.dft(array, cv2_dft_array)

plt.plot(cv2_dft_array)
plt.show()

# pi2 = np.pi * 2.0


def my_dft(input, output):
    for i in range(0, input.shape[0]):
        sum = 0
        for j in range(0, input.shape[0]):
            e = np.exp(-1j * np.pi * i * j / input.shape[0])
            sum += e * input[j]
        output[i] = sum
    return output


my_dft_array = np.zeros((1000), 'float32')

my_dft_array = my_dft(array, my_dft_array)

plt.plot(my_dft_array)
print(my_dft_array[0:5])
print(cv2_dft_array[0:5])
plt.show()































