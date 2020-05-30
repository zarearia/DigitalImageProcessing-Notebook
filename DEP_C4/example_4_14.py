import numpy as np
import matplotlib.pyplot as plt
import cv2


# TODO: The Box
box_img = cv2.imread('/Users/ariazare/Projects/Python/DEP_C4/the_box.png')
box_img = cv2.cvtColor(box_img, cv2.COLOR_RGB2GRAY)

box_complex = [np.array(box_img, 'float32'), np.zeros(box_img.shape, 'float32')]
box_complex = cv2.merge(box_complex)
box_complex = cv2.dft(box_complex, box_complex)
box_real = box_complex[0:600, 0:600, 0]
box_imag = box_complex[0:600, 0:600, 1]

box_magnitude = np.sqrt(np.square(box_real) + np.square(box_imag))
box_phase = np.arctan2(box_real, box_imag)


# TODO: The House
house_img = cv2.imread('/Users/ariazare/Projects/Python/DEP_C4/Fig0438(a)(bld_600by600).tif')
house_img = cv2.cvtColor(house_img, cv2.COLOR_RGB2GRAY)

house_complex = [np.array(house_img, 'float32'), np.zeros(house_img.shape, 'float32')]
house_complex = cv2.merge(house_complex)
house_complex = cv2.dft(house_complex, house_complex)
house_real = house_complex[0:house_complex.shape[0], 0:house_complex.shape[1], 0]
house_imag = house_complex[0:house_complex.shape[0], 0:house_complex.shape[1], 1]

house_magnitude = np.sqrt(np.square(house_real) + np.square(house_imag))
house_phase = np.arctan2(house_real, house_imag)


# TODO: Inverse Fourier Transform House(Phase Only)

house_complex_phase_fourier_transform = np.exp(1j * house_phase)
house_complex_phase_rebuild = [np.array(house_complex_phase_fourier_transform.imag, 'float32'),
                               np.array(house_complex_phase_fourier_transform.real, 'float32')]
house_complex_phase_rebuild = cv2.merge(house_complex_phase_rebuild)
cv2.idft(house_complex_phase_rebuild, house_complex_phase_rebuild)

# plt.imshow(house_complex_phase_rebuild[:,:,0], 'Greys')
# plt.show()


# TODO: Inverse Fourier Transform House(Magnitude Only)

house_complex_mag_fourier_transform = np.exp(1j * 1) * house_magnitude
house_complex_mag_rebuild = [np.array(house_complex_mag_fourier_transform.imag, 'float32'),
                             np.array(house_complex_mag_fourier_transform.real, 'float32')]
house_complex_mag_rebuild = cv2.merge(house_complex_mag_rebuild)
cv2.idft(house_complex_mag_rebuild, house_complex_mag_rebuild)

# plt.imshow(house_complex_mag_rebuild[:,:,0], 'Greys')
# plt.show()


# TODO: Inverse Fourier Transform House_Phase & Box_Mag

housey_box_complex_fourier_transform = np.exp(1j * house_phase) * box_magnitude
housey_box_complex_rebuild = [np.array(housey_box_complex_fourier_transform.imag, 'float32'),
                              np.array(housey_box_complex_fourier_transform.real, 'float32')]
housey_box_complex_rebuild = cv2.merge(housey_box_complex_rebuild)
cv2.idft(housey_box_complex_rebuild, housey_box_complex_rebuild)

# plt.imshow(housey_box_complex_rebuild[:,:,0], cmap='Greys')
# plt.show()


# TODO: Inverse Fourier Transform House_Mag & Box_House

boxy_house_complex_fourier_transform = np.exp(1j * box_phase) * house_magnitude
boxy_house_complex_rebuild = [np.array(boxy_house_complex_fourier_transform.imag, 'float32'),
                              np.array(boxy_house_complex_fourier_transform.real, 'float32')]
boxy_house_complex_rebuild = cv2.merge(boxy_house_complex_rebuild)
cv2.idft(boxy_house_complex_rebuild, boxy_house_complex_rebuild)

# plt.imshow(boxy_house_complex_rebuild[:, :, 1])
# plt.show()


# TODO: Inverse Fourier Transform """"Test""""

# # house_complex_phase_only = [np.array(house_phase, 'float32'), np.zeros(house_phase.shape, 'float32')]
# # house_complex_phase_only = cv2.merge(house_complex_phase_only)
# # print(house_complex_phase_only.shape)
# #
# # cv2.idft(house_complex_phase_only, house_complex_phase_only)
# #
# # plt.imshow(house_complex_phase_only[0:600,0:600,0], cmap='Greys')
# # plt.show()
#
# house_complex_rebuild_complex = np.exp(1j * house_phase) * house_magnitude
# print(house_complex_rebuild_complex.shape)
# house_complex_rebuild = [np.array(house_complex_rebuild_complex.imag, 'float32'),
#                          np.array(house_complex_rebuild_complex.real, 'float32')]
# house_complex_rebuild = cv2.merge(house_complex_rebuild)
# # house_complex_rebuild[:,:,0] = house_complex_rebuild_complex[:,:].real
# # house_complex_rebuild[:,:,1] = house_complex_rebuild_complex[:,:].imag
# # print(house_complex_rebuild)
# # house_complex_phase_only = [house_complex_rebuild, house_complex_rebuild]
# # house_complex_phase_only = cv2.merge(house_complex_phase_only)
#
#
# cv2.idft(house_complex_rebuild, house_complex_rebuild)
# plt.imshow(house_complex_rebuild[:, :,0])
# plt.show()
#
# if np.equal(house_img.all(), house_complex_rebuild[:,:,0].all()):
#     print('OK')
#
# cv2.imshow("test", house_complex_rebuild[:,:,0])
#
#
# cv2.waitKey(0)

















