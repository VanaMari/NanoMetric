# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 19:55:51 2024

@author: zenob
"""

# import os
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from random import sample, choice, randint
from tensorflow.keras import Input, Model
# from skimage.transform import resize
# from tensorflow.keras import backend as K
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.layers import Conv2D, LeakyReLU, Concatenate, Dropout, UpSampling2D, BatchNormalization
# from sklearn.metrics import jaccard_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
# from sklearn.metrics import confusion_matrix, jaccard_score, accuracy_score, precision_score, recall_score, f1_score
# from glob import glob
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.metrics import MeanIoU
# from sklearn.metrics import jaccard_score
# from sklearn.model_selection import KFold
# from skimage.io import imread
# import pandas as pd
# import seaborn as sns
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.models import load_model

# Network
def U_Net256(img_shape=(256, 256, 3), gf=64, output_activation="sigmoid"):
    """U-Net Generator"""

    def conv2d(layer_input, filters, f_size=4, bn=True):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)  #alpha  usar negative_slope para o Colab
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = BatchNormalization(momentum=0.8)(u)
        u = Concatenate()([u, skip_input])
        return u

    # Image input
    d0 = Input(shape=img_shape)

    # Downsampling
    d1 = conv2d(d0, gf, bn=False)
    d2 = conv2d(d1, gf*2)
    d3 = conv2d(d2, gf*4)
    d4 = conv2d(d3, gf*8)
    d5 = conv2d(d4, gf*8)
    d6 = conv2d(d5, gf*8)
    d7 = conv2d(d6, gf*8)

    # Upsampling
    u1 = deconv2d(d7, d6, gf*8)
    u2 = deconv2d(u1, d5, gf*8)
    u3 = deconv2d(u2, d4, gf*8)
    u4 = deconv2d(u3, d3, gf*4)
    u5 = deconv2d(u4, d2, gf*2)
    u6 = deconv2d(u5, d1, gf)

    u7 = UpSampling2D(size=2)(u6)
    output_img = Conv2D(1, kernel_size=4, strides=1, padding='same', activation=output_activation)(u7)

    return Model(d0, output_img)