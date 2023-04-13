import deeplake
import numpy
import tensorflow as tf
from matplotlib import pyplot as plt

ds = deeplake.load('hub://udayuppal/kuzushiji-kanji')

print('--------data loaded--------')


img = ds.images[0:100].numpy()


# plt.imshow(img, interpolation='nearest')
# plt.show()

dataloader = ds.tensorflow()

iterator = iter(dataloader)
while(True):
    print(iterator.get_next())
