import cv2
import numpy as np
img = cv2.imread("test2.png", 0)

#dilate
kernel = np.ones((3,3),np.uint8)
img = cv2.erode(img,kernel,iterations = 2)
img = cv2.dilate(img, kernel, iterations=1)
cv2.imwrite('test2out.png', img)