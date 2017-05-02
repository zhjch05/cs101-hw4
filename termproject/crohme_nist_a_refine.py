import numpy as np
import cv2

a = np.load('a32.npz')
a_x = a['x']
a_y = a['y']
a_x = a_x.astype('float32')
a_x = a_x.reshape(a_y.shape[0], 32, 32, 1)
print('loaded a')
crohme = np.load('crohme32.npz')
crohme_x = crohme['x']
crohme_y = crohme['y']
crohme_x = crohme_x.astype('float32')
crohme_x = crohme_x.reshape(crohme_y.shape[0], 32, 32, 1)
print('loaded crohme')
nist = np.load('NIST32.npz')
nist_x = nist['x']
nist_y = nist['y']
nist_x = nist_x.astype('float32')
nist_x = nist_x.reshape(nist_y.shape[0], 32, 32, 1)
print('loaded nist')
x = np.append(nist_x, a_x)
x = np.append(x, crohme_x)
y = np.append(nist_y, a_y)
y = np.append(y, crohme_y)

np.savez_compressed('final32.npz', x=x, y=y)
