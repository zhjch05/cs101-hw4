import numpy as np
import cv2
from multiprocessing.dummy import Pool as ThreadPool
pool = ThreadPool(12)

crohme = np.load('A.npz')
x = crohme['x']
y = crohme['y']
x = x.astype('float32')
x = x.reshape(y.shape[0], 128, 128, 3)
print('loaded')
cv2.imshow('img', x[1])
cv2.waitKey(0)


def resize(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)


new_x = pool.map(resize, x)

np.savez_compressed('a32.npz', x=new_x, y=y)
