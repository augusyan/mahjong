# -*- coding: utf-8 -*-
# @Time    : 18-2-27 下午4:39
# @Author  : Yan
# @Site    : 
# @File    : sys_model_test0.py
# @Software: PyCharm Community Edition
# @Function: keras achieve mahjong model
# @update:
"""
利用keras对tensorflow的代码进行重构的简单版
一般的epoch=10
根据层数不同命名
"""
from __future__ import division
from __future__ import print_function

import keras
import numpy as np
import matplotlib.pyplot as plt
import datetime

np.random.seed(1337)  # for reproducibility
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
from keras import regularizers
from keras.utils.vis_utils import plot_model


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
        plt.savefig('loss_u1200_e10_c1_l2r_t0.png')
        plt.show()


# time record
starttime = datetime.datetime.now()

batch = 50
input_features = 296
output_features = 34
units = 1200
num_epoch = 30
model_save_path = 'sys_model_u1200_e10_c1_l2r_t0.h5'
model_pic = 'model_u1200_e10_c1_l2r_t0.png'

# data path
trainData_tmp = np.loadtxt('sys_kings_train_5.txt', delimiter=' ', dtype=np.float16)
# trainData_tmp = np.loadtxt('dataset_v11_train_test.txt', delimiter=' ', dtype=np.float16)
testData_tmp = np.loadtxt('sys_kings_test_5.txt', delimiter=' ', dtype=np.float16)
# testData_tmp = np.loadtxt('dataset_v11_mini_test.txt', delimiter=' ', dtype=np.float16)

x_train = np.array(trainData_tmp[:, 0:-1])
y_train = np.array(trainData_tmp[:, -1]).astype(np.int32)
y_train = np_utils.to_categorical(y_train, num_classes=34)

x_test = np.array(testData_tmp[:, 0:-1])
y_test = np.array(testData_tmp[:, -1]).astype(np.int32)
y_test = np_utils.to_categorical(y_test, num_classes=34)


# print(np.shape(x_train))
# print(np.shape(y_train))
# print(np.shape(x_test))
# print(np.shape(y_test))


# model define
model = Sequential()
model.add(Dense(units, input_dim=input_features, activation='relu'))
# model.add(Convolution2D(batch_input_shape=(None, 1, 1, input_features),
#                         filters=128,
#                         kernel_size=1,
#                         strides=1,
#                         padding='same',
#                         data_format='channels_first',))
# model.add(Activation('relu'))
# model.add(Flatten())
model.add(Dense(units, activation='relu'))
# model.add(Dropout(0.1))
model.add(Dense(units,activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(units, activation='relu'))
model.add(Dense(output_features))
model.add(Dense(output_features,input_dim=output_features,kernel_regularizer=regularizers.l2(0.001),
                activity_regularizer=regularizers.l1(0.001)))
model.add(Activation('softmax'))


adam = Adam(lr=1e-4)
# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
print(model)

# 创建一个实例history
history = LossHistory()
plot_model(model, to_file=model_pic)

# 开始训练和测试
print('Training ------------')
model.fit(x_train, y_train, batch_size=batch,epochs=num_epoch, validation_data=(x_test, y_test), callbacks=[history])
print('\nTesting ------------')
loss, accuracy = model.evaluate(x_test, y_test)
print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)
model.save(model_save_path)
# 绘制acc-loss曲线
history.loss_plot('epoch')

endtime = datetime.datetime.now()
print('usetime | ',endtime - starttime)

