import numpy as np
import cv2
from multiprocessing.dummy import Pool as ThreadPool
pool = ThreadPool(12)

crohme = np.load('crohme_compressed.npz')
x = crohme['x']
y = crohme['y']
x = x.reshape(y.shape[0], 135, 135, 1)

def resize(img):
    return cv2.resize(img, (32, 32), interpolation = cv2.INTER_AREA)
new_x = pool.map(resize, x)

np.savez_compressed('crohme32.npz', x=new_x, y=y)
