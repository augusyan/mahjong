# -*- coding: utf-8 -*-
# @Time    : 18-4-25 上午11:06
# @Author  : Yan
# @Site    : 
# @File    : run_basic_train.py
# @Software: PyCharm Community Edition
# @Function: a basic train file import models, + resnet_simple, +
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
        plt.savefig('loss/loss_sys_resnet_simple_e10d5.png')
        # plt.show()


def model_test(model_path, X):
    """
    Test model
    :param model_path:
    :param X:
    :return:
    """
    model = load_model(model_path)  # 载入模型
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
model_save_path = 'model/sys_resnet_simple_e10d5.h5'  # 模型保存地址
model_pic = 'model/sys_resnet_simple_e10d5.png'  # 绘图保存地址
res_block = 7  # 残差块
log = './log/sys_resnet_simple_e10.logd5'  # log 地址
model_restore = 'model/sys_resnet_simple_e10d5.h5'
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

# model

a = Resnet_v1()
model = a.build()

plot_model(model, to_file='Resnet_simple_v1.png', show_shapes=True)
adam = Adam(lr=1e-4)
# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
print(model)

# #载入模型
# model = load_model(model_restore)
# print('Model Restore!')
# 创建一个实例history

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
