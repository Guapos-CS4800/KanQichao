# matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline

# os
import os

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import resample,shuffle
from sklearn.preprocessing import LabelEncoder
# NumPy
import numpy as np
from numpy import savez_compressed

# pandas
import pandas as pd

# Path
import pathlib 


import sys

# Images
from PIL import Image, ImageOps

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential




my_dir = "kanji_dataset"
# Dummy black image/label to setup the ndarray
imgs = np.zeros((64,64), np.uint8).reshape(1,64,64) 
labels = np.array(['XXX'])

for item in pathlib.Path(my_dir).glob('**/*.jpg'):
  image = np.array(Image.open(item)).reshape(1,64,64)
  

  imgs = np.concatenate([imgs, image])
  parent = os.path.dirname(item).split('/')[-1]
  labels = np.concatenate([labels,np.array([parent])])
  
# Delete the dummy picture
imgs = np.delete(imgs,0,0)
labels = np.delete(labels,0,0)# Save as npz file
np.savez_compressed('content/kkanji-imgs.npz', imgs)
np.savez_compressed('content/kkanji-labels.npz', labels)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
def load(f):
    return np.load(f, allow_pickle=True)['arr_0']



X = load('content/kkanji-imgs.npz')
Y = load('content/kkanji-labels.npz') 



# print(f"Training Images: {X.shape}")

# print(f"Training Labels: {Y.shape}")


    
# # # testImage = 1451
# # Image.fromarray(X[testImage])

# # # unicodevalue = Y[testImage]
# # print(unicodevalue[16:22])

# splitting the data in half (half in testing half in training)
testSize = 0.5

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=testSize, random_state=1, stratify=Y)

lb = LabelEncoder()
y_test = lb.fit_transform(y_test)
y_train = lb.fit_transform(y_train)

# print(X[0].shape)


# print(f"Training Images: {x_train.shape}")
# print(f"Training Labels: {y_train.shape}")
# print(f"Test Images: {x_test.shape}")
# print(f"Test Labels: {y_test.shape}")

# print(f"Unique Kanji characters (Training): \t {np.unique(y_train).size}")
# print(f"Unique Kanji characters (Test): \t {np.unique(y_test).size}")


batch_size = 32
img_height = 64
img_width = 64

train_ds = tf.keras.utils.image_dataset_from_directory(
'kanji_dataset',
color_mode='grayscale',
validation_split=0.2,
subset="training",
seed=123,
image_size=(img_height, img_width),
batch_size=batch_size,)

val_ds = tf.keras.utils.image_dataset_from_directory(
'kanji_dataset',
validation_split=0.2,
color_mode='grayscale',
subset="validation",
seed=123,
image_size=(img_height, img_width),
batch_size=batch_size,)

#class_names = train_ds.class_names

folder = './kanji_dataset'

class_names = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# # data_augmentation = keras.Sequential(
# #   [
# #     layers.RandomFlip("horizontal",
# #                       input_shape=(img_height,
# #                                   img_width,
# #                                   3)),
# #     layers.RandomRotation(0.1),
# #     layers.RandomZoom(0.1),
# #   ]
# # )
# # reconstructed_model = keras.models.load_model("saved_model")

input_shape = (64, 64, 1)
model = Sequential([
    layers.Rescaling(1./255, input_shape=input_shape),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(.2),
    layers.Dense(128, activation='relu'),
    layers.Dense(100)
])


model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
model.summary()

epochs=5
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

model.save('my_model.h5')