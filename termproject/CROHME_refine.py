import numpy as np
import cv2
import glob
import sys

x = np.empty((0))
y = np.empty((0))

#class names to extract
# '0','2','3','4','5','6','7','8','9', 'X','y','A','b','c','d','m','n','p', 'f','H','k','l', Use NIST first and then try to add CROHME's
categories = ['Delta','pi','+','-', '=', 'div','times','sqrt','(',')']

#dataset path
path = './dataset/extracted_images/'

def format(img):
	#padding to 90 * 90
	padding = 90 - img.shape[0]
	img = np.lib.pad(img, ((padding, padding),
		(padding, padding)), 'constant',
		constant_values=((255, 255), (255, 255)))

	#threshold and invert color
	ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

	#dilate
	kernel = np.ones((3,3),np.uint8)
	img = cv2.dilate(img,kernel,iterations = 2)

	img = cv2.GaussianBlur(img, (5, 5), 0)
	ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

	return img

t = 0
for label in categories:
	category_path = path + label + '/'
	c_pics = glob.glob(category_path + '*')

	#samples size for debug
	print(label, len(c_pics))

	i = 0
	l = len(c_pics)
	for pic in c_pics:
		if i > 5000:
			break
		if pic.endswith('.png') or pic.endswith('.jpg'):

			#read in
			img = cv2.imread(pic, 0)

			img = format(img)

			#write for debug
			# cv2.imwrite('./output/' + str(t) + '.png', img)

			img = img/255

			x = np.append(x, img.flatten())
			y = np.append(y, label)

			p = i*100.0/min(l, 5000)
			sys.stdout.write("\r%d%%" % p)
			sys.stdout.flush()
			i+=1
			t+=1
	print("%s done." % label)

#save to file
np.savez('crohme_refined.npz', x = x, y = y)