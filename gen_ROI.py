import cv2
import numpy as np
import os

img = cv2.imread("view8/0810/v0810_nbfixcameramove(3)/1/1.bmp")

h, w = img.shape[:2]

for index, (i, j) in enumerate([(i, j) for i in range(3) for j in range(3)]):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    roi_start = ( (w*j - 200) // (3), (h*i - 200) // (3) )
    roi_end = ( (w*(j+1) + 200) // (3), (h*(i+1) + 200) // (3) )

    cv2.rectangle(mask, roi_start, roi_end, 255, -1)
    result = cv2.bitwise_and(img, img, mask=mask)

    if not os.path.exists('_ROI'):
        os.makedirs('_ROI')

    cv2.imwrite(f'_ROI/roi{index+1}.bmp', result)
# os.remove('roi5.bmp')
