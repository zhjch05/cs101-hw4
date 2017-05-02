# Random Shifts
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from keras import backend as K
import numpy as np
K.set_image_dim_ordering('tf')
# load data
crohme = np.load('crohme_compressed.npz')
x = crohme['x']
y = crohme['y']
# reshape to be [samples][pixels][width][height]
x = x.reshape(y.shape[0], 135, 135, 1)
# convert from int to float
x = x.astype('float32')
# define data preparation
shift = 0.2
datagen = ImageDataGenerator(width_shift_range=shift, height_shift_range=shift)
# fit parameters from data
datagen.fit(x)
# configure batch size and retrieve one batch of images
for X_batch, y_batch in datagen.flow(x, y, batch_size=9):
	# create a grid of 3x3 images
	for i in range(0, 9):
		pyplot.subplot(330 + 1 + i)
		pyplot.imshow(X_batch[i].reshape(135, 135), cmap=pyplot.get_cmap('gray'))
	# show the plot
	pyplot.show()
	break
