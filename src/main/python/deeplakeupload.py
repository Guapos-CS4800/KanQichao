import deeplake
import numpy
import tensorflow as tf
from matplotlib import pyplot as plt

ds = deeplake.load('hub://udayuppal/kuzushiji-kanji')

print('--------data loaded--------')