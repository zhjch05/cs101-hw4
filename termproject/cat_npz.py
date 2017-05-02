import numpy as np
import glob
import sys
import os
gx = np.empty((0))
gy = np.empty((0))
path = os.path.join(os.getcwd(), 'dataset','npz')
print(path)
npzs = glob.glob(path + '\\*')
for npz in npzs:
	print(npz)
	n = np.load(npz)
	print(npz, 'loaded')
	x = n['x']
	y = n['y']
	gx = np.append(gx, x)
	gy = np.append(gy, y)

np.savez_compressed('NIST128.npz', x=gx, y=gy)
