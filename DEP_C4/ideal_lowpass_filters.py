import numpy as np
import matplotlib.pyplot as plt
import cv2
import utilities


# MARK: the image

img = cv2.imread('/Users/ariazare/Projects/Python/DEP_C4/Fig0441(a)(characters_test_pattern).tif')
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

img_padded = np.zeros((img.shape[0] * 2, img.shape[1] * 2), 'float32')
img_padded[:img.shape[0], :img.shape[1]] = img

img_padded = utilities.center_frequency(img_padded)

complex_img = [img_padded, np.zeros(img_padded.shape, 'float32')]
complex_img = cv2.merge(complex_img)
cv2.dft(complex_img, complex_img)

# img_mag = utilities.get_magnitude(complex_img)
# img_phase = utilities.get_phase_angel(complex_img)
#
# img_mag = cv2.log(img_mag)
# plt.imshow(img_mag, 'Greys')
# plt.show()
#
# total_image_power = np.sum(img_phase)


# MARK: the filter

# also exist in utilities
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


ideal_lowpass_filter = utilities.creat_ideal_lowpass_filter((img_padded.shape), 160)
ideal_lowpass_filter = [ideal_lowpass_filter, ideal_lowpass_filter]
ideal_lowpass_filter = cv2.merge(ideal_lowpass_filter)

filtered_img = np.copy(complex_img)
filtered_img = complex_img * ideal_lowpass_filter
cv2.idft(filtered_img, filtered_img)
filtered_img = filtered_img[:img.shape[0], :img.shape[1], 0]
filtered_img = utilities.center_frequency(filtered_img)
plt.imshow(filtered_img, 'Greys')
plt.show()





















