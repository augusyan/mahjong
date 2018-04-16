# -*- coding: utf-8 -*-
# @Time    : 18-3-28 上午10:46
# @Author  : Yan
# @Site    : 
# @File    : TwinRes_sys2_nonkings_v0.1.py
# @Software: PyCharm Community Edition
# @Function: 
# @update:A demo for twin-resnet approach


import keras
import datetime
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils
from keras.layers import add, Flatten,Dropout
import numpy as np
import matplotlib.pyplot as plt
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D, AveragePooling2D,MaxPooling2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import load_model
from keras.applications.resnet50 import ResNet50
from keras import regularizers
from keras.models import Sequential
from plot_loss_keras import LossHistory

seed = 7
np.random.seed(seed)

model = Sequential()
model.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=(227, 227, 3), padding='valid', activation='relu',
                 kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.summary()


