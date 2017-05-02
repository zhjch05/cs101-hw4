import numpy as np
import cv2
crohme = np.load('crohme32.npz')
x = crohme['x']
y = crohme['y']
x = x.reshape(y.shape[0], 32, 32, 1)
cv2.imshow('img', x[0])
cv2.waitKey(0)
