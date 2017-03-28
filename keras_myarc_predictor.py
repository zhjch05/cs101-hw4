from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import model_from_json

import sys
import glob
import skimage.transform
import skimage.io
import tensorflow as tf
import ntpath
from skimage.color import rgb2grey
import numpy as np

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
        output_file.write("%s\t%i\n" % (ntpath.basename(all_pics[i]),tf.argmax(loaded_model.predict(image),1).eval(session=sess)[0]))
        i+=1
    output_file.close()
else:
    print("Please specify a folder name .. exiting")
    quit()