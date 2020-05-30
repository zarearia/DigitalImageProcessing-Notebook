import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('/Users/ariazare/Projects/Python/DEP_C5/test6.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

img = cv2.GaussianBlur(img, (5, 5), 0)

img = cv2.medianBlur(img, 5)

img2 = np.zeros(img.shape, 'float32')

for i in range(5, img.shape[0] - 5):
    for j in range(5, img.shape[1] - 5):
        nums = np.zeros((5, 5), 'float32')
        for k in range(-2, 2):
            for l in range(-2, 2):
                nums[k + 2, l + 2] = img[i + k, j + l]
        if nums[0, 0] < 170:
            nums[0, 0] = 0
        else:
            
            nums[0, 0] = 255
        img2[i, j] = nums[0, 0]

# for i in range(5, img.shape[0] - 5):
#     for j in range(5, img.shape[1] - 5):
#         if img2[i, j] < 170:
#             img2[i, j] = 0
#         else:
#             img2[i, j] = 255

img2 = cv2.GaussianBlur(img2, (5, 5), 0)

img2 = cv2.GaussianBlur(img2, (7, 7), 0)

plt.imshow(img2, 'Greys_r')
plt.show()




