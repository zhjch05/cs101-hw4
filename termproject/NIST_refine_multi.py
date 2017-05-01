import numpy as np
import cv2
import glob
import sys
import skimage.io
from multiprocessing import Process, Array

imgs = {}
n_imgs = {}

#class names to extract
categories = ['0','1','2','3','4','5','6','7','8','9',
	'x','y','A','a','b','c','d',
	'm','n','p','f','h','k']

c_map = {
	'0':'30',
	'1':'31',
	'2':'32',
	'3':'33',
	'4':'34',
	'5':'35',
	'6':'36',
	'7':'37',
	'8':'38',
	'9':'39',
	'A':'41',
	'a':'61',
	'b':'62',
	'c':'63',
	'd':'64',
	'h':'68',
	'k':'6b',
	'f':'66',
	'm':'6d',
	'n':'6e',
	'p':'70',
	'x':'78',
	'y':'79'
}

def format(img):
	img = cv2.GaussianBlur(img, (3, 3), 0)
	ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
	return img

def readin(imgs, label, x, y):
	global n_imgs
	x = np.empty((0))
	y = np.empty((0))
	i = 0
	pics = imgs[label]
	for img in pics:
		if i > 10:
			break

		# img = format(img)

		#write for debug
		# cv2.imwrite('./output/' + str(t) + '.png', img)

		img = img/255

		x = np.append(x, img.flatten())
		y = np.append(y, label)
		print(i, label)
		i+=1
	print("%s done." % label)

#dataset path
path = './dataset/by_class/'

#read in all pics to mem at one time

for label in categories:
	c_path = path + c_map[label] + '/train_' + c_map[label] + '/'
	c_pics = glob.glob(c_path + '*')

	#samples size for debug
	print(label, len(c_pics))

	imgs[label] = skimage.io.imread_collection(c_pics)

for label in imgs:
	# readin(imgs, label)
	shared_x = Array('x', [])
	shared_y = Array('y', [])
	p = Process(target=readin, args=(imgs, label, shared_x, shared_y))
	p.start()
	p.join()



	

#save to file
# np.savez_compressed('nist_refined.npz', x = x, y = y)