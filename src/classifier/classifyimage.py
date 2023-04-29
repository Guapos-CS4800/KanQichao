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




# # my_dir = " classifier/kanji_dataset"
# # Dummy black image/label to setup the ndarray
# # imgs = np.zeros((64,64), np.uint8).reshape(1,64,64) 
# # labels = np.array(['XXX'])

# # for item in pathlib.Path(my_dir).glob('**/*.jpg'):
# #   image = np.array(Image.open(item)).reshape(1,64,64)
  

# #   imgs = np.concatenate([imgs, image])
# #   parent = os.path.dirname(item).split('/')[-1]
# #   labels = np.concatenate([labels,np.array([parent])])
  
# # Delete the dummy picture
# # imgs = np.delete(imgs,0,0)
# # labels = np.delete(labels,0,0)# Save as npz file
# # np.savez_compressed(' classifier/content/kkanji-imgs.npz', imgs)
# # np.savez_compressed(' classifier/content/kkanji-labels.npz', labels)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
def main():

  reconstructed_model = tf.keras.models.load_model("classifier/my_model.h5")


  # acc = history.history['accuracy']
  # val_acc = history.history['val_accuracy']

  # loss = history.history['loss']
  # val_loss = history.history['val_loss']

  # epochs_range = range(epochs)

  # plt.figure(figsize=(8, 8))
  # plt.subplot(1, 2, 1)
  # plt.plot(epochs_range, acc, label='Training Accuracy')
  # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  # plt.legend(loc='lower right')
  # plt.title('Training and Validation Accuracy')

  # plt.subplot(1, 2, 2)
  # plt.plot(epochs_range, loss, label='Training Loss')
  # plt.plot(epochs_range, val_loss, label='Validation Loss')
  # plt.legend(loc='upper right')
  # plt.title('Training and Validation Loss')
  # plt.show()

  for i in range(1):
    # print('start')  

    sunflower_path = './sample.jpg'
    # ./tempkanjitotest/U+56DB/a20bccf447c10059.png
    img = tf.keras.utils.load_img(
      sunflower_path,
      color_mode='grayscale',
      target_size=((64, 64)),
      interpolation='nearest',
      keep_aspect_ratio=False

    )

    # img.show()
    

    img_array = tf.keras.utils.img_to_array(img)
  
    x = np.unique(img_array)
    # print('this is x, ', x)
    # print(img_array)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    # print('love', img_array.shape)


    predictions = reconstructed_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    folder = './classifier/kanji_dataset'
    class_names = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
    
    print('---------------------')
    print("Possible Classes:")
    fileCharacters = open("fileCharacters.txt", "w", encoding='utf-8')
    for name in class_names:
        fileCharacters.write(name)
        fileCharacters.write("\n")
    fileCharacters.close()
    print('---------------------')

    # print(
    #     "This image most likely belongs to {} with a {:.2f} percent confidence."
    #     .format(class_names[np.argmax(score)], 100 * np.max(score))
    # )

    temp = np.argpartition(-score, 10)
    result_args = temp[:10]

    temp = np.partition(-score, 10)
    result = -temp[:10]

    confList = []
    for index, conf in zip(result_args, result):
      confList.append((conf, index))

    print('---------------------')  
    values = sorted(confList,key=lambda x: x[0], reverse=True)
    print(values)
    print('---------------------')

    topTen = ''
    for conf, index in values:
      topTen += class_names[index] + ' '
    
    with open("classifier/predictedkanji.txt", "w", encoding='utf-8') as f:
      print(topTen, file=f)
      

  # score = reconstructed_model.evaluate(x_test, y_test,  verbose = 0) 
  # print('Test loss:', score[0]) 
  # print('Test accuracy:', score[1])

  # stringyup = class_names[3]
  # testingunicode = ('\\' + stringyup)
  # print(testingunicode)
  # t = testingunicode.encode('utf-8').decode('unicode-escape')
  # print(t)
