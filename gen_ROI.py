import cv2
import numpy as np
import os
import glob

images = glob.glob('_ROI/*.bmp')

h, w = cv2.imread(images[0]).shape[:2]

for idx, fname in enumerate(images):
    results = []
    img = cv2.imread(fname)
    for i, j in [(i, j) for i in range(3) for j in range(3)]:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        roi_start = ( (w*j - 300) // (3), (h*i - 300) // (3) )
        roi_end = ( (w*(j+1) + 300) // (3), (h*(i+1) + 300) // (3) )

        cv2.rectangle(mask, roi_start, roi_end, 255, -1)
        result = cv2.bitwise_and(img, img, mask=mask)
        results.append(result)
        if not os.path.exists('_ROI'):
            os.makedirs('_ROI')

    cv2.imwrite(f'_ROI/roi{idx}.bmp', results[idx])
