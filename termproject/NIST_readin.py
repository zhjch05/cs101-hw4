import numpy as np
import cv2
import glob
import sys
import os
import skimage.io

imgs = {}
x = np.empty((0))
y = np.empty((0))

# class names to extract
categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
	'x', 'y', 'A', 'a', 'b', 'c', 'd',
	'm', 'n', 'p', 'f', 'h', 'k']

c_map = {
	'0': '30',
	'1': '31',
	'2': '32',
	'3': '33',
	'4': '34',
	'5': '35',
	'6': '36',
	'7': '37',
	'8': '38',
	'9': '39',
	'A': '41',
	'a': '61',
	'b': '62',
	'c': '63',
	'd': '64',
	'h': '68',
	'k': '6b',
	'f': '66',
	'm': '6d',
	'n': '6e',
	'p': '70',
	'x': '78',
	'y': '79'
}

# dataset path
path = os.path.join(os.getcwd(), 'dataset', 'by_class', 'by_class')

# read in all pics to mem at one time

for label in categories:
    c_path = os.path.join(path, c_map[label], 'train_' + c_map[label])
    c_pics = glob.glob(c_path + '\\*')

    # samples size for debug
    print(label, len(c_pics))

    imgs[label] = skimage.io.imread_collection(c_pics)
    this_y = [label] * len(imgs[label])
    this_x = np.empty((0))
    for img in imgs[label]:
        this_x = np.append(this_x, img.flatten())
    x = np.append(x, this_x)
    y = np.append(y, this_y)

np.savez_compressed('nistreadin.npz', x=x, y=y)
