import numpy as np
import cv2
import glob
import sys

#class names to extract
categories = ['0','1','2','3','4','5','6','7','8','9',
	'x','y','A','a','b','c','d',
	'm','n','p','f','h','k','Delta','pi','+','-', '=', 'div','times','sqrt','(',')']

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

def readin(c_pics):
	global x, y
	i = 0
	l = len(c_pics)
	t = min(l, 2000)
	for pic in c_pics:
		if i > 2000:
			break
		if pic.endswith('.png') or pic.endswith('.jpg'):

			#read in
			img = cv2.imread(pic, 0)

			# img = format(img)

			#write for debug
			# cv2.imwrite('./output/' + str(t) + '.png', img)

			img = img/255

			x = np.append(x, img.flatten())
			y = np.append(y, label)

			p = i*100.0/t
			sys.stdout.write("\r%d%%" % p)
			sys.stdout.flush()
			i+=1
	print("%s done." % label)

#dataset path
path = './dataset/by_class/'

x = np.empty((0))
y = np.empty((0))

for label in categories:
	c_path = path + c_map[label] + '/train_' + c_map[label] + '/'
	c_pics = glob.glob(c_path + '*')

	#samples size for debug
	print(label, len(c_pics))

	readin(c_pics)
	

#save to file
np.savez_compressed('nist_refined.npz', x = x, y = y)