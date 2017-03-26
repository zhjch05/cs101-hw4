from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import sys
import glob
import skimage.transform
import skimage.io
import tensorflow as tf
import ntpath
from skimage.color import rgb2grey
import numpy as np

# Building convolutional network
network = input_data(shape=[None, 28, 28, 1], name='input')
network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 1024, activation='tanh')
network = dropout(network, 1.0)
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=1e-4,
                     loss='categorical_crossentropy', name='target')

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)

model.load('hw4_20.tflearn')

sess = tf.Session()

if(len(sys.argv)>1):
    folder_name = sys.argv[1]
    all_pics = glob.glob("./"+folder_name+"/*")
    images = skimage.io.imread_collection(all_pics)
    i=0
    output_file = open("prediction.txt", "w")
    for image in images:
        # image = rgb2grey(image) # cast rgb to grayscale
        # image = 255 - image # invert color
        image = skimage.transform.resize(image,(28,28))
        # skimage.io.imsave("./output/"+ntpath.basename(all_pics[i])+"_output.png",arr=image) # save resized images to visualize them
        image = image.reshape([-1, 28, 28, 1])
        #print(ntpath.basename(all_pics[i]),"\t",tf.argmax(model.predict(image),1).eval(session=sess)[0]) # test print
        output_file.write("%s\t%i\n" % (ntpath.basename(all_pics[i]),tf.argmax(model.predict(image),1).eval(session=sess)[0]))
        i+=1
    output_file.close()
else:
    print("Please specify a folder name .. exiting")
    quit()