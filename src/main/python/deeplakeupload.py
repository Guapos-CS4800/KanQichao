import deeplake
import numpy
import tensorflow as tf
from matplotlib import pyplot as plt

import sys



def loadDeeplakeSet(out):
    ds = deeplake.load('hub://udayuppal/kuzushiji-kanji')

    out.write('--------data loaded--------')