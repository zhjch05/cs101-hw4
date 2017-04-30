import numpy as np
import cv2
img = cv2.imread('41_2.png', 0)
img = cv2.resize(img, (32, 32), interpolation = cv2.INTER_AREA)
img = cv2.GaussianBlur(img, (3, 3), 0)
ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
cv2.imwrite('41_2out.png', img)