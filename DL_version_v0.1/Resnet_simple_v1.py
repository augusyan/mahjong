# -*- coding: utf-8 -*-
# @Time    : 18-4-25 上午10:31
# @Author  : Yan
# @Site    : 
# @File    : Resnet_simple_v1.py
# @Software: PyCharm Community Edition
# @Function: define 3-ways input, InceptionV3 resnet blocks
# @update:

import keras
import datetime
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils
from keras.layers import add
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.engine.topology import Input, Layer
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D, AveragePooling2D, MaxPooling2D, UpSampling2D, Cropping2D
from keras.layers.core import Activation, Dense, Flatten, Dropout
from keras.layers.merge import Add, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import load_model, Sequential
from keras import regularizers
from keras.layers.core import Lambda


class Resnet_v1(object):
    """
    define namespace, 3main repeat network, each has 1 input conv-3, 3 resnetblock with direct connect, 1 fcn
    resnetblock has a conv-2,

    """

    def __init__(self, num_blocks=3, res_layer_num=3, l2_reg=0.001):
        # self.x = inpt
        self.num_blocks = num_blocks
        self.res_layer_num = res_layer_num
        self.l2_reg = l2_reg

    def slice1(self, x, slide):
        """
        Lambda split the feature maps into 2 pieces
        :param x:
        :param slide:
        :return:
        """
        x1 = x[:, :slide, :, :]
        return x1

    def slice2(self, x, slide):
        x1 = x[:, slide:, :, :]
        return x1

    def Input_Conv(self, x):
        """
        Copy the weights and do Conv
        :param x: Input
        :return: x
        """

        x = BatchNormalization(axis=1, name="input_batchnorm1-1")(x)
        x1 = x2 = x3 = x

        x1 = Conv2D(filters=64, kernel_size=3, padding="same",
                    data_format="channels_first", use_bias=False, kernel_regularizer=l2(self.l2_reg),
                    name='input_conv1-1')(x1)

        x2 = Conv2D(filters=64, kernel_size=3, padding="same",
                    data_format="channels_first", use_bias=False, kernel_regularizer=l2(self.l2_reg),
                    name='input_conv1-2')(x2)

        x3 = Conv2D(filters=64, kernel_size=3, padding="same",
                    data_format="channels_first", use_bias=False, kernel_regularizer=l2(self.l2_reg),
                    name='input_conv1-3')(x3)

        x = Concatenate(axis=1,name='input_Concatenate1')([x1, x2, x3])

        x = BatchNormalization(axis=1,name='input_batchnorm2-1')(x)
        x = Conv2D(filters=192, kernel_size=1, padding="same",
                        data_format="channels_first", use_bias=False, kernel_regularizer=l2(self.l2_reg),
                   name='input_conv2-1')(x)

        x = Activation("relu",name='input_relu1')(x)
        return x

    def Inception_blocks(self, x, nb_filter=64, num_res=0):
        """

        :param x:
        :param nb_filter:
        :param num_res:
        :return:
        """
        in_x = x
        x = Conv2D(filters=192, kernel_size=3, padding='same', strides=1, data_format="channels_first",
                           name='inception_conv_'+ str(num_res))(x)
        x = BatchNormalization(axis=1, name='inception_conv_bn_' + str(num_res))(x)

        branch1x1 = Conv2D(filters=nb_filter, kernel_size=1, padding='same', strides=1, data_format="channels_first",
                           name='branch1x1_'+ str(num_res))(x)
        branch1x1 = Activation("relu", name='branch1x1_relu'+str(num_res))(branch1x1)
        branch1x1 = BatchNormalization(axis=1,name='branch1x1_bn_'+str(num_res))(branch1x1)

        branch3x3 = Conv2D(filters=nb_filter,kernel_size=1, padding='same', strides=1, data_format="channels_first",
                           name='branch3x3-1_'+str(num_res))(x)
        branch3x3 = Activation("relu", name='branch3x3_relu-1_'+str(num_res))(branch3x3)
        branch3x3 = BatchNormalization(axis=1,name='branch3x3_bn-1_'+str(num_res))(branch3x3)
        branch3x3 = Conv2D(filters=nb_filter, kernel_size=3, padding='same', strides=1,data_format="channels_first",
                           name='branch3x3-2_'+str(num_res))(branch3x3)
        branch3x3 = Activation("relu", name='branch3x3_relu-2_' + str(num_res))(branch3x3)
        branch3x3 = BatchNormalization(axis=1, name='branch3x3_bn-2_' + str(num_res))(branch3x3)

        # branch5x5 = Conv2D(filters=nb_filter, kernel_size=1, padding='same', strides=1,name='branch3x3-1_' + str(num_res))(x)
        # branch5x5 = Activation("relu", name='branch5x5_relu-1_' + str(num_res))(branch5x5)
        # branch5x5 = BatchNormalization(axis=1, name='branch5x5_bn-1_' + str(num_res))(branch5x5)
        # branch5x5 = Conv2D(filters=nb_filter, kernel_size=5, padding='same', strides=1,
        #                    name='branch5x5-2_' + str(num_res))(branch5x5)
        # branch5x5 = Activation("relu", name='branch5x5_relu-2_' + str(num_res))(branch5x5)
        # branch5x5 = BatchNormalization(axis=1, name='branch5x5_bn-2_' + str(num_res))(branch5x5)

        branchpool = MaxPooling2D(data_format="channels_first", pool_size=3, strides=1, padding='same',name='branchpool_'+str(num_res))(x)
        branchpool = Conv2D(filters=nb_filter, kernel_size=1, padding='same', strides=1,data_format="channels_first",
                                    name='branchpool_conv_'+str(num_res))(branchpool)
        branchpool = BatchNormalization(axis=1, name='branchpool_bn_' + str(num_res))(branchpool)

        x = Concatenate(axis=1,name='Concatenate_'+str(num_res))([branch1x1, branch3x3, branchpool])

        x = Add(name="add_"+str(num_res))([in_x, x])

        return x

    def build(self):
        """
        Builds the full Keras model and stores it in self.model.
        """
        in_x = x = Input((1, 19, 19))
        for i in range(20):
            x = self.Inception_blocks(x, 64, i+1)
        x = AveragePooling2D(pool_size=(7, 7))(x)
        x = Flatten()(x)
        # x = Dense(1024, activation='softmax')(x)
        x = Dense(34, activation='softmax')(x)
        model = Model(inputs=in_x, outputs=x)
        model.summary()

        # x_test = np.array([[[1,2],[2,3],[3,4],[4,5]]])
        # print (model.predict(x_test))
        plot_model(model, to_file='Resnet_simple_v1.png', show_shapes=True)
        return model


a = Resnet_v1()
model=a.build()
