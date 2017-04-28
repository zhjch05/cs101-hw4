import numpy as np
import cv2
import glob
import sys

x = np.empty((0))
y = np.empty((0))

#class names to extract
categories = ['0','2','3','4','5','6','7','8','9','x','y','A','b','c','d','m','n','p','Delta',
	'f','H','k','l','pi','+','-', '=', 'div','times','sqrt','(',')']

#dataset path
path = './dataset/extracted_images/'