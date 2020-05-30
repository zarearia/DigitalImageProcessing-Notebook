import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)


# Mark: reading and converting img to value between 0 and 1

img = cv.imread('/Users/ariazare/Projects/Python/DEP_C4/Fig0438(a)(bld_600by600).tif')

img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

img = np.array(img, 'float32')
# img /= 255


# Mark: creating RGB sin funcs


def creat_sin_func(array, multiple, plus=0):
    for i in range(0, array.shape[0]):
        array[i] = np.abs(np.sin((i/255) * np.pi * multiple + plus))
    return array


blue_func = np.zeros((256), 'float32')
blue_func = creat_sin_func(blue_func, 1)

red_func = np.zeros((256), 'float32')
red_func = creat_sin_func(red_func, 1, 1)

green_func = np.zeros((256), 'float32')
green_func = creat_sin_func(green_func, 1, 2)


# Mark: creating RGB image

def mix_gray_image_with_color_sin_finc(img, color_func):
    img_l = np.copy(img)
    num = np.array([0], 'int')
    for i in range(0, img_l.shape[0]):
        for j in range(0, img_l.shape[1]):
            num[0] = img_l[i, j]
            img_l[i, j] = color_func[num[0]]
    return img_l


blue_only_img = np.zeros(img.shape, 'int')
blue_only_img = mix_gray_image_with_color_sin_finc(img, blue_func)

green_only_img = np.zeros(img.shape, 'int')
green_only_img = mix_gray_image_with_color_sin_finc(img, green_func)

red_only_img = np.zeros(img.shape, 'int')
red_only_img = mix_gray_image_with_color_sin_finc(img, red_func)

color_img = np.zeros((img.shape[0], img.shape[1], 3))
color_img[:, :, 0] = red_only_img
color_img[:, :, 1] = green_only_img
color_img[:, :, 2] = blue_only_img

plt.imshow(img)
plt.show()

plt.imshow(color_img)
plt.show()

































