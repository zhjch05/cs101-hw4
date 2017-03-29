from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import model_from_json
import numpy as np

import sys
import glob
import skimage.transform
import skimage.io
import tensorflow as tf
import ntpath
from skimage.color import rgb2grey
import numpy as np
import numpy.lib

def add_padding(image):
    if image.shape[0] > image.shape[1]:
        l_edge = image.shape[0]
        diff = image.shape[0] - image.shape[1]
        front = int(diff/2)
        rear = diff - front
        new_image = np.lib.pad(image, ((0, 0), (front, rear)), 'constant', constant_values=((0, 0),))
        return new_image
    elif image.shape[0] < image.shape[1]:
        l_edge = image.shape[1]
        diff = image.shape[1] - image.shape[0]
        front = int(diff / 2)
        rear = diff - front
        new_image = np.lib.pad(image, ((front, rear), (0, 0)), 'constant', constant_values=((0, 0),))
        return new_image
    else:
        return image

def watershed_inner(x, threshold):
    if x > threshold:
        return 1
    else:
        return 0


def watershed(image, threshold):
    new_image = np.empty(image.shape)
    for i in range(len(image)):
        for j in range(len(image[i])):
            new_image[i][j] = watershed_inner(image[i][j], threshold)
    return new_image

sess = tf.Session()

# load json and create model
json_file = open('model_2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_2.h5")
print("Loaded model from disk")
loaded_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

test = {}
correct = 0
#read in annotation (test)
with open("annotation.txt","r") as annotations:
    for line in annotations:
        image, label = line.split()
        test[image] = label
total = len(test)

if(len(sys.argv)>1):
    folder_name = sys.argv[1]
    all_pics = glob.glob("./"+folder_name+"/*")
    images = skimage.io.imread_collection(all_pics)
    output_file = open("prediction.txt", "w")
    i=0
    for image in images:
        # image = rgb2grey(image) # cast rgb to grayscale
        # image = 255 - image # invert color
        image = add_padding(image)
        image = skimage.transform.resize(image,(28,28))
        # image = watershed(image, wst)
        # skimage.io.imsave("./output/"+ntpath.basename(all_pics[i]),arr=image) # save resized images to visualize them
        image = image.reshape([-1, 28, 28, 1])
        #print(ntpath.basename(all_pics[i]),"\t",tf.argmax(model.predict(image),1).eval(session=sess)[0]) # test print
        output_file.write("%s\t%i\n" % (ntpath.basename(all_pics[i]),tf.argmax(loaded_model.predict(image),1).eval(session=sess)[0]))
        i+=1
    output_file.close()
else:
    print("Please specify a folder name .. exiting")
    quit()