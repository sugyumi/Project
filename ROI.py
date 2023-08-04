import cv2
import numpy as np
import os

img = cv2.imread("v0804_110(3)/a/7.bmp")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ROI
h, w = img.shape[:2]
# h, w = gray.shape[:2]

for index, (i, j) in enumerate([(i, j) for i in range(3) for j in range(3)]):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    # mask = np.zeros(gray.shape[:2], dtype=np.uint8)
    roi_start = (w*j // (3), h*i// (3))
    roi_end = (w*(j+1) // (3), h*(i+1) // (3))

    cv2.rectangle(mask, roi_start, roi_end, 255, -1)
    result = cv2.bitwise_and(img, img, mask=mask)
    # result = cv2.bitwise_and(gray, gray, mask=mask)

    cv2.imwrite(f'roi{index+1}.bmp', result)
# os.remove('roi5.bmp')