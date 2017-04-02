from __future__ import print_function
import keras
from keras.models import model_from_json

import sys
import glob
import skimage.io
import tensorflow as tf
import ntpath
import numpy as np
import cv2

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

rate = 5.0

def aug_inner(x, rate):
    auged = x * rate
    if auged > 1.0000:
        return 1.0
    else:
        return auged

def aug_signal(image, rate):
    new_image = np.empty(image.shape)
    for i in range(len(image)):
        for j in range(len(image[i])):
            new_image[i][j] = aug_inner(image[i][j], rate)
    return new_image

sess = tf.Session()

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
loaded_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

if(len(sys.argv)>1):
    folder_name = sys.argv[1]
    all_pics = glob.glob("./"+folder_name+"/*")
    images = skimage.io.imread_collection(all_pics)
    output_file = open("prediction.txt", "w")
    i=0
    for image in images:
        image = add_padding(image)
        image = cv2.resize(image, (28, 28), interpolation = cv2.INTER_AREA)
        image = image/255.0
        image = aug_signal(image, rate)
        image = image.reshape([-1, 28, 28, 1])
        output_file.write("%s\t%i\n" % (ntpath.basename(all_pics[i]),tf.argmax(loaded_model.predict(image),1).eval(session=sess)[0]))
        i+=1
    output_file.close()
else:
    print("Please specify a folder name .. exiting")
    quit()