# -*- coding: utf-8 -*-
# @Time    : 18-3-23 上午11:01
# @Author  : Yan
# @Site    : 
# @File    : Res1001_sys_nonkings_v2.py.py
# @Software: PyCharm Community Edition
# @Function: 
# @update:

import keras
import datetime
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils
from keras.layers import add, Flatten,Dropout
import numpy as np
import matplotlib.pyplot as plt
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import load_model
from keras.applications.resnet50 import ResNet50
from keras import regularizers
from keras.models import Sequential

seed = 7
np.random.seed(seed)


# 写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.savefig('loss/sys_e40_res1001.png')
        plt.show()


def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x


def build(self):
    """
    Builds the full Keras model and stores it in self.model.
    """
    mc = self.config.model
    in_x = x = Input((18, 8, 8))

    # (batch, channels, height, width)
    x = Conv2D(filters=256, kernel_size=3, padding="same",
               data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg),
               name="input_conv-" + str(mc.cnn_first_filter_size) + "-" + str(mc.cnn_filter_num))(x)
    x = BatchNormalization(axis=1, name="input_batchnorm")(x)
    x = Activation("relu", name="input_relu")(x)

    for i in range(mc.res_layer_num):
        x = self._build_residual_block(x, i + 1)

    res_out = x

    # # for policy output
    # x = Conv2D(filters=2, kernel_size=1, data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg),
    #            name="policy_conv-1-2")(res_out)
    # x = BatchNormalization(axis=1, name="policy_batchnorm")(x)
    # x = Activation("relu", name="policy_relu")(x)
    # x = Flatten(name="policy_flatten")(x)
    # # no output for 'pass'
    # policy_out = Dense(self.config.n_labels, kernel_regularizer=l2(mc.l2_reg), activation="softmax", name="policy_out")(
    #     x)
    #
    # # for value output
    # x = Conv2D(filters=4, kernel_size=1, data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg),
    #            name="value_conv-1-4")(res_out)
    # x = BatchNormalization(axis=1, name="value_batchnorm")(x)
    # x = Activation("relu", name="value_relu")(x)
    # x = Flatten(name="value_flatten")(x)
    # x = Dense(mc.value_fc_size, kernel_regularizer=l2(mc.l2_reg), activation="relu", name="value_dense")(x)
    # value_out = Dense(1, kernel_regularizer=l2(mc.l2_reg), activation="tanh", name="value_out")(x)
    #
    # self.model = Model(in_x, [policy_out, value_out], name="chess_model")


def _build_residual_block( x, index):
    # mc = self.config.model
    in_x = x
    res_name = "res" + str(index)
    x = BatchNormalization(axis=1, name=res_name + "_batchnorm1")(x)
    x = Activation("relu", name=res_name + "_relu1")(x)
    x = Conv2D(filters=256, kernel_size=3, padding="same",
               data_format="channels_first", use_bias=False, kernel_regularizer=l2(0.001),
               name=res_name + "_conv1-" + str(3) + "-" + str(3))(x)
    x = BatchNormalization(axis=1, name=res_name + "_batchnorm2")(x)
    x = Activation("relu", name=res_name + "_relu2")(x)
    x = Conv2D(filters=256, kernel_size=3, padding="same",
               data_format="channels_first", use_bias=False, kernel_regularizer=l2(0.001),
               name=res_name + "_conv2-" + str(3) + "-" + str(3))(x)
    x = Add(name=res_name + "_add")([in_x, x])
    # x = Activation("relu", name=res_name + "_relu2")(x)
    return x


def model_test(model_path,X):
    """
    Test model
    :param model_path:
    :param X:
    :return:
    """
    model = load_model(model_path)   # 载入模型
    print('Model Restore!')
    print(model.predict_classes(X, batch_size=1, verbose=0))

# time record
starttime = datetime.datetime.now()

# global define
batch = 64  # 批次大小
input_features = 361  # 数据的维数
output_features = 34  # 输出类别
units = 1024  # fc神经元数
num_epoch = 10  # 训练轮数
model_save_path = 'model/sys_e40_res1001.h5'  # 模型保存地址
model_pic = 'model/sys_e40_res1001.png'  # 绘图保存地址
res_block = 7   # 残差块
log = './log/sys_e40_res1001.log'  #log 地址
model_restore = 'model/sys_e20_res1001.h5'
# data path
x_train = np.loadtxt('mj_data/non_king_processed/X_train_non_king_1_pic.txt', delimiter=' ', dtype=np.float16)
x_test = np.loadtxt('mj_data/non_king_processed/X_test_non_king_1_pic.txt', delimiter=' ', dtype=np.int32)
y_train = np.loadtxt('mj_data/non_king_processed/Y_train_non_king_1.txt', delimiter=' ', dtype=np.float16)
y_test = np.loadtxt('mj_data/non_king_processed/Y_test_non_king_1.txt', delimiter=' ', dtype=np.int32)

y_train = np_utils.to_categorical(y_train, num_classes=34)
y_test = np_utils.to_categorical(y_test, num_classes=34)

# reshape
x_train = x_train.reshape(-1, 1, 19, 19)
x_test = x_test.reshape(-1, 1, 19, 19)

# def Conv_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
#     x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
#     x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
#     if with_conv_shortcut:
#         shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
#         x = add([x, shortcut])
#         return x
#     else:
#         x = add([x, inpt])
#         return x


# inpt = Input(shape=(224, 224, 3))
# x = ZeroPadding2D((3, 3))(inpt)
# x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
# x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
# # (56,56,64)
# x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
# x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
# x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
# # (28,28,128)
# x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
# x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
# x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
# x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
# # (14,14,256)
# x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
# x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
# x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
# x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
# x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
# x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
# # (7,7,512)
# x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
# x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
# x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
# x = AveragePooling2D(pool_size=(7, 7))(x)
# x = Flatten()(x)
# x = Dense(1000, activation='softmax')(x)


in_x = x = Input((1, 19, 19))
# (batch, channels, height, width)
x = Conv2D(filters=256, kernel_size=5, padding="same",
               data_format="channels_first", use_bias=False, kernel_regularizer=l2(0.001),
               name="input_conv-" + str(5) + "-" + str(256))(x)
x = BatchNormalization(axis=1, name="input_batchnorm")(x)
x = Activation("relu", name="input_relu")(x)

for i in range(20):
    x = _build_residual_block(x, i + 1)
res_out = x

x = AveragePooling2D(pool_size=(7, 7))(x)
x = Flatten()(x)
# x = Dense(1024, activation='softmax')(x)
x = Dense(34, activation='softmax')(x)

model = Model(inputs=in_x, outputs=x)
# model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# model.summary()

plot_model(model, to_file='model_ResNet1001.png',show_shapes=True)
adam = Adam(lr=1e-4)
# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
print(model)

#载入模型
model = load_model(model_restore)
print('Model Restore!')
#创建一个实例history

history = LossHistory()
plot_model(model, to_file=model_pic)

# 开始训练和测试
print('Training ------------')
hist_log = model.fit(x_train, y_train, batch_size=batch, epochs=num_epoch, validation_data=(x_test, y_test),
                     callbacks=[history])
print('Saving Log -------------')
with open(log, 'w') as f:
    f.write(str(hist_log.history))
print('\nTesting ------------')
loss, accuracy = model.evaluate(x_test, y_test)
print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)
model.save(model_save_path)
# 绘制acc-loss曲线
history.loss_plot('epoch')

endtime = datetime.datetime.now()
print('usetime | ', endtime - starttime)
