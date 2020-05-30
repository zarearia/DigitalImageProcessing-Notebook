import numpy as np
import cv2


def rearrange_frequency_spectrum(array):
    first_quart = array[0:array.shape[0]//2, 0:array.shape[1]//2]
    second_quart = array[array.shape[0]//2:array.shape[0], 0:array.shape[1]//2]
    third_quart = array[0:array.shape[0]//2, array.shape[1]//2:array.shape[1]]
    fourth_quart = array[array.shape[0]//2:array.shape[0], array.shape[1]//2:array.shape[1]]

    first_quart_back = np.copy(first_quart)
    first_quart = fourth_quart
    fourth_quart = first_quart_back
    second_quart_back = np.copy(second_quart)
    second_quart = third_quart
    third_quart = second_quart_back

    array[0:array.shape[0]//2, 0:array.shape[1]//2] = first_quart
    array[array.shape[0]//2:array.shape[0], 0:array.shape[1]//2] = second_quart
    array[0:array.shape[0]//2, array.shape[1]//2:array.shape[1]] = third_quart
    array[array.shape[0]//2:array.shape[0], array.shape[1]//2:array.shape[1]] = fourth_quart

    return array


def center_frequency(array):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            array[i, j] *= np.power(-1, i + j)

    return array


def get_fourier_transform(array):
    complex_array = [np.array(array, 'float32'), np.zeros(array.shape, 'float32')]
    complex_array = cv2.merge(complex_array)
    cv2.dft(complex_array, complex_array)
    complex_array_real = complex_array[:, :, 0]
    complex_array_imag = complex_array[:, :, 1]

    return (complex_array, complex_array_real, complex_array_imag)


def get_magnitude(array):
    magnitude = np.sqrt(np.square(array[:, :, 0]) + np.square(array[:, :, 1]))
    return magnitude


def get_phase_angel(array):
    phase_angel = np.arctan2(array[:, :, 0], array[:, :, 1])
    return phase_angel


def merge_mag_phase(mag, phase):
    complex_array = np.multiply(mag, np.exp(1j * phase))
    array = np.zeros((mag.shape[0], mag.shape[0], 2), 'float32')
    array[:, :, 0] = complex_array.real
    array[:, :, 1] = complex_array.imag
    return array


def creat_ideal_lowpass_filter(full_shape, circle_radius):
    ideal_lowpass_filter = np.zeros(full_shape, 'float32')
    ideal_lowpass_filter_p = ideal_lowpass_filter.shape[0]
    ideal_lowpass_filter_q = ideal_lowpass_filter.shape[1]
    for i in range(ideal_lowpass_filter_p // 2 - circle_radius + 1, ideal_lowpass_filter_p // 2 + circle_radius + 1):
        for j in range(ideal_lowpass_filter_q // 2 - circle_radius + 1, ideal_lowpass_filter_q // 2 + circle_radius + 1):
            if (np.sqrt(np.square(i - ideal_lowpass_filter_p // 2)
                        + np.square(j - ideal_lowpass_filter_q // 2)) <= circle_radius):
                ideal_lowpass_filter[i, j] = 1
    return ideal_lowpass_filter


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
















