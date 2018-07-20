# -*- coding: utf-8 -*-
# @Time    : 18-6-19 上午2:39
# @Author  : Yan
# @Site    : 
# @File    : run_model.py
# @Software: PyCharm Community Edition
# @Function: A complete run model function for DL train and test
# @update:

import keras
import datetime
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils
from keras.layers import add, Flatten, Dropout
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
from Resnet_simple_v1 import Resnet_v1
from keras.callbacks import TensorBoard

from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
from keras.applications import *

import os


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
        plt.savefig('loss/20180528/loss_sys_non_king_op_resnet_simple_e50d5.png')
        # plt.show()


# class ConfigPic(object):
#     batch = 64  # 批次大小
#     input_features = 361  # 数据的维数
#     output_features = 34  # 输出类别
#     units = 1024  # fc神经元数
#     num_epoch = 10  # 训练轮数
#     model_save_path = 'model/20180528/sys_king_resnet_simple_e0-10d5.h5'  # 模型保存地址
#     model_pic = 'model/20180528/sys_king_resnet_simple_e0-10d5.png'  # 绘图保存地址
#     res_block = 7  # 残差块
#     log = './log/20180528/sys_king_resnet_simple_e0-10d5.log'  # log 地址
#     # model_restore = 'model/20180528/sys_king_resnet_simple_e0-10d5.h5'
#     # data path
#     x_train = np.loadtxt('mj_data/king_20180528/output_X_train_pic.txt', delimiter=' ', dtype=np.float16)
#     x_test = np.loadtxt('mj_data/king_20180528/output_X_test_pic.txt', delimiter=' ', dtype=np.int32)
#     y_train = np.loadtxt('mj_data/king_20180528/output_Y_train.txt', delimiter=' ', dtype=np.float16)
#     y_test = np.loadtxt('mj_data/king_20180528/output_Y_test.txt', delimiter=' ', dtype=np.int32)
#
#     y_train = np_utils.to_categorical(y_train, num_classes=34)
#     y_test = np_utils.to_categorical(y_test, num_classes=34)
#
#     # reshape
#     x_train = x_train.reshape(-1, 1, 19, 19)
#     x_test = x_test.reshape(-1, 1, 19, 19)
#
#
# class ConfigSen(object):
#     batch = 64  # 批次大小
#     input_features = 361  # 数据的维数
#     output_features = 34  # 输出类别
#     units = 1024  # fc神经元数
#     num_epoch = 1  # 训练轮数
#     model_save_path = 'model/sys_resnet_simple_e10-20d5.h5'  # 模型保存地址
#     model_pic = 'model/sys_resnet_simple_e10-20d5.png'  # 绘图保存地址
#     res_block = 7  # 残差块
#     log = './log/sys_resnet_simple_e10-20d5.log'  # log 地址
#     model_restore = 'model/sys_resnet_simple_e10d5.h5'
#     # data path
#     x_train = np.loadtxt('mj_data/non_king_processed/X_train_non_king_1_pic.txt', delimiter=' ', dtype=np.float16)
#     x_test = np.loadtxt('mj_data/non_king_processed/X_test_non_king_1_pic.txt', delimiter=' ', dtype=np.int32)
#     y_train = np.loadtxt('mj_data/non_king_processed/Y_train_non_king_1.txt', delimiter=' ', dtype=np.float16)
#     y_test = np.loadtxt('mj_data/non_king_processed/Y_test_non_king_1.txt', delimiter=' ', dtype=np.int32)
#
#     y_train = np_utils.to_categorical(y_train, num_classes=34)
#     y_test = np_utils.to_categorical(y_test, num_classes=34)
#
#     # reshape
#     x_train = x_train.reshape(-1, 1, 19, 19)
#     x_test = x_test.reshape(-1, 1, 19, 19)

def model_train(model_path, X):
    pass


def model_test(model_path, target):
    """
    Test model
    :param model_path:
    :param X:
    :return:
    """
    # 忽略硬件加速的警告信息
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # file_path = 'images/0a70f64352edfef4c82c22015f0e3a20.jpg'
    #
    # img = image.load_img(file_path, target_size=(224, 224))
    # x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    #
    model = load_model(model_path)
    y = model.predict(target)
    # print(np.argmax(y))

    # print('Predicted:', decode_predictions(y, top=3)[0])
    model = load_model(model_path)  # 载入模型
    print('Model Restore!')
    print(model.predict_classes(y, batch_size=1, verbose=0))


# time record
starttime = datetime.datetime.now()

# global define
batch = 32  # 批次大小
input_features = 256  # 数据的维数
output_features = 8  # 输出类别
units = 1024  # fc神经元数
num_epoch = 50  # 训练轮数
model_save_path = 'model/20180528/sys_non_king_op_resnet_simple_e50d5.h5'  # 模型保存地址
model_pic = 'model/20180528/sys_non_king_op_resnet_simple_e50d5.png'  # 绘图保存地址
res_block = 7  # 残差块
log = './log/20180528/sys_non_king_op_resnet_simple_e50d5.log'  # log 地址
# model_restore = 'model/20180528/sys_king_resnet_simple_e0-10d5.h5'
# data path
x_train = np.loadtxt('mj_data/data_201804/non_king_op_20180528/output_X_train_pic.txt', delimiter=' ', dtype=np.float16)
x_test = np.loadtxt('mj_data/data_201804/non_king_op_20180528/output_X_test_pic.txt', delimiter=' ', dtype=np.int32)
y_train = np.loadtxt('mj_data/data_201804/non_king_op_20180528/output_Y_train.txt', delimiter=' ', dtype=np.float16)
y_test = np.loadtxt('mj_data/data_201804/non_king_op_20180528/output_Y_test.txt', delimiter=' ', dtype=np.int32)

y_train = np_utils.to_categorical(y_train, num_classes=8)
y_test = np_utils.to_categorical(y_test, num_classes=8)

# reshape
x_train = x_train.reshape(-1, 1, 16, 16)
x_test = x_test.reshape(-1, 1, 16, 16)

# model

a = Resnet_v1()
model = a.build()

plot_model(model, to_file='Resnet_simple_v3.png', show_shapes=True)
adam = Adam(lr=1e-4)
# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
print(model)

# 载入模型
#model = load_model(model_restore)
#print('Model Restore!')
# 创建一个实例history

# history = LossHistory()
plot_model(model, to_file=model_pic)

# 开始训练和测试
print('Training ------------')
hist_log = model.fit(x_train, y_train, batch_size=batch, epochs=num_epoch, validation_data=(x_test, y_test),
                     callbacks=[TensorBoard(log_dir='log/')])
# hist_log = model.fit(x_train, y_train, batch_size=batch, epochs=num_epoch, validation_data=(x_test, y_test),
#                      callbacks=[history])
print('Saving Log -------------')
with open(log, 'w') as f:
    f.write(str(hist_log.history))
print('\nTesting ------------')
loss, accuracy = model.evaluate(x_test, y_test)
print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)
model.save(model_save_path)
# 绘制acc-loss曲线
# history.loss_plot('epoch')

endtime = datetime.datetime.now()
print('usetime | ', endtime - starttime)
