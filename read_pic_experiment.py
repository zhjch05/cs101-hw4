import skimage.io

import matplotlib.pyplot as plt

import numpy as np

file_path="./test_folder/images.png"
image = skimage.io.imread(file_path)


imgplot = plt.imshow(image)