import numpy as np 
import os
import sys
import tensorflow as tf

from PIL import Image
from matplotlib import pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

#import convert


batch_size = 128
num_classes = 10
epochs = 12
img_rows, img_cols = 28, 28

def load(f):
	return np.load(f)['arr_0']


def loadImages(out):
    #load data
    print("Loading Images")
    x_train = load('kmnist-train-imgs.npz')
    x_test = load('kmnist-test-imgs.npz')
    y_train = load('kmnist-train-labels.npz')
    y_test = load('kmnist-test-labels.npz')


    out.write('{} train samples, {} test samples'.format(len(x_train), len(x_test)))