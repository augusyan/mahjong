# -*- coding: utf-8 -*-
# @Time    : 18-4-12 下午11:25
# @Author  : Yan
# @Site    : 
# @File    : google_block_noname.py
# @Software: PyCharm Community Edition
# @Function: 
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


class GoRes_Model(object):
    """
    lalala

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

    def Input_Conv(self, x, num_res=0):
        """
        Copy the weights and do Conv
        :param x: Input
        :return: x
        """

        x = BatchNormalization(axis=1, name="input_batchnorm1-1" + 'Block' + str(num_res))(x)
        x1 = x2 = x3 = x

        x1 = Conv2D(filters=64, kernel_size=3, padding="valid",
                    data_format="channels_first", use_bias=False, kernel_regularizer=l2(self.l2_reg))(x1)


        x2 = Conv2D(filters=64, kernel_size=3, padding="valid",
                    data_format="channels_first", use_bias=False, kernel_regularizer=l2(self.l2_reg))(x2)

        x3 = Conv2D(filters=64, kernel_size=3, padding="valid",
                    data_format="channels_first", use_bias=False, kernel_regularizer=l2(self.l2_reg))(x3)

        x = Concatenate(axis=1)([x1, x2, x3])

        x = BatchNormalization(axis=1)(x)
        x = Conv2D(filters=192, kernel_size=1, padding="same",
                        data_format="channels_first", use_bias=False, kernel_regularizer=l2(self.l2_reg))(x)

        x = Activation("relu")(x)
        return x

    def Conv2d_BN(self, x, nb_filter, kernel_size, padding='same', strides=1, name='', num_res=0):
        """

        :param x:
        :param nb_filter:
        :param kernel_size:
        :param padding:
        :param strides:
        :param name:
        :return:
        """
        conv_name = "conv-" + name
        bn_name = 'bn-' + name

        x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, data_format="channels_first",
                   activation='relu')(x)
        x = BatchNormalization(axis=1)(x)
        return x

    def Inception(self, x, nb_filter, layer=0, num_res=0):
        name = str(layer) + '-Inception-'
        branch1x1 = self.Conv2d_BN(x, nb_filter, 1, padding='same', strides=1, num_res=num_res)

        branch3x3 = self.Conv2d_BN(x, nb_filter, 1, padding='same', strides=1, num_res=num_res)
        branch3x3 = self.Conv2d_BN(branch3x3, nb_filter, 3, padding='same', strides=1,
                                   num_res=num_res)

        branch5x5 = self.Conv2d_BN(x, nb_filter, 1, padding='same', strides=1, num_res=num_res)
        branch5x5 = self.Conv2d_BN(branch5x5, nb_filter, 5, padding='same', strides=1,
                                   num_res=num_res)

        branchpool = MaxPooling2D(data_format="channels_first", pool_size=3, strides=1, padding='same')(x)
        branchpool = self.Conv2d_BN(branchpool, nb_filter, 1, padding='same', strides=1, name=name + str(6),
                                    num_res=num_res)

        x = Concatenate(axis=1)([branch1x1, branch3x3, branch5x5, branchpool])

        return x

    def double_Conv_Block(self, x, f_map, f_map_out, num_dCR, layer):
        dCB = 'dCB_'
        slide = int(f_map / 2)  # split the input feature maps
        x1 = Lambda(self.slice1, arguments={'slide': slide})(x)
        x2 = Lambda(self.slice2, arguments={'slide': slide})(x)

        x1 = BatchNormalization(axis=1, name='dCR-batchnorm' + num_dCR + str(layer))(x1)
        x1 = Conv2D(filters=slide, kernel_size=3, padding="valid",
                    data_format="channels_first", use_bias=False, kernel_regularizer=l2(self.l2_reg)
                    )(x1)

        x2 = BatchNormalization(axis=1)(x2)
        x2 = Conv2D(filters=slide, kernel_size=3, padding="valid",
                    data_format="channels_first", use_bias=False, kernel_regularizer=l2(self.l2_reg)
                    )(x2)

        output = Concatenate(axis=1)([x1, x2])

        output = BatchNormalization(axis=1)(output)
        print(np.shape(output))
        output = Conv2D(filters=f_map, kernel_size=1, padding="same",
                        data_format="channels_first", use_bias=False, kernel_regularizer=l2(self.l2_reg)
                        )(output)
        output = Activation("relu")(output)

        output = self.Inception(output, (int(f_map_out / 2)), layer + 1)

        return output

    def GoRes_blocks(self, x, num_res):
        _name = 'GoRes_B_' + str(num_res)
        inpt_x = x
        x = self.Input_Conv(x, num_res=num_res)
        x = self.double_Conv_Block(x, 192, 256, _name,layer=0)
        x = self.double_Conv_Block(x, 256, 256, _name,1)
        x = self.double_Conv_Block(x, 256, 256, _name,2)
        x = UpSampling2D(size=(2, 2), data_format='channels_first')(x)
        x = Conv2D(filters=256, kernel_size=4, padding="valid",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(self.l2_reg)
                   )(x)
        x = Conv2D(filters=1, kernel_size=1, padding="valid",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(self.l2_reg)
                   )(x)

        x = Add()([inpt_x, x])
        x = Activation("relu")(x)
        return x

    def build(self):
        """
        Builds the full Keras model and stores it in self.model.
        """
        in_x = x = Input((1, 19, 19))
        x = self.GoRes_blocks(x, 1)
        x = self.GoRes_blocks(x, 2)
        x = self.GoRes_blocks(x, 3)
        x = Flatten()(x)
        x = Dense(34, activation='softmax')(x)
        model = Model(inputs=in_x, outputs=x)
        model.summary()

        # x_test = np.array([[[1,2],[2,3],[3,4],[4,5]]])
        # print (model.predict(x_test))
        plot_model(model, to_file='lambda1.png', show_shapes=True)
        return model


        # (batch, channels, height, width)
        # for i in range(self.res_layer_num):
        #     x = self._build_residual_block(x, i + 1)
        #
        # res_out = x

        # # for policy output
        # x = Conv2D(filters=2, kernel_size=1, data_format="channels_first", use_bias=False,
        #            kernel_regularizer=l2(mc.l2_reg),
        #            name="policy_conv-1-2")(res_out)
        # x = BatchNormalization(axis=1, name="policy_batchnorm")(x)
        # x = Activation("relu", name="policy_relu")(x)
        # x = Flatten(name="policy_flatten")(x)
        # # no output for 'pass'
        # policy_out = Dense(self.config.n_labels, kernel_regularizer=l2(mc.l2_reg), activation="softmax",
        #                    name="policy_out")(
        #     x)
        #
        # # for value output
        # x = Conv2D(filters=4, kernel_size=1, data_format="channels_first", use_bias=False,
        #            kernel_regularizer=l2(mc.l2_reg),
        #            name="value_conv-1-4")(res_out)
        # x = BatchNormalization(axis=1, name="value_batchnorm")(x)
        # x = Activation("relu", name="value_relu")(x)
        # x = Flatten(name="value_flatten")(x)
        # x = Dense(mc.value_fc_size, kernel_regularizer=l2(mc.l2_reg), activation="relu", name="value_dense")(x)
        # value_out = Dense(1, kernel_regularizer=l2(mc.l2_reg), activation="tanh", name="value_out")(x)

        # self.model = Model(in_x, [policy_out, value_out], name="chess_model")

#
# a = GoRes_Model()
# a.build()
