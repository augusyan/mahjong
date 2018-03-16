# -*- coding: utf-8 -*-
# @Time    : 18-3-6 下午3:26
# @Author  : Yan
# @Site    : 
# @File    : DNNmodel_sys_nonkings_v2.py
# @Software: PyCharm Community Edition
# @Function: 
# @update:
"""
利用keras对tensorflow的代码进行重构的简单版,数据集是zhiyi_v2版本的
全连接版的模型
根据层数不同命名,规则如下
e10--epoch=10,c1--cov=1,l2r--regularizers=l2,
d2--dense=2,u1200--units=1200,con--continue train
dp5--dropout=0.5
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
from keras.models import load_model


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
        plt.savefig('loss_sysv2_u1200e100d3dp5.png')  # loss-acc图保存
        plt.show()


# time record
starttime = datetime.datetime.now()

# global define
batch = 50  # 批次大小
input_features = 329  # 数据的维数
output_features = 34  # 输出类别
units = 1200  # fc神经元数
num_epoch = 100  # 训练轮数
model_save_path = 'model/sysv2_u1200e100d3dp5.h5'  # 模型保存地址
# model_restore = 'model/sysv2_u1200e10d3l2r.h5' # 模型调用地址
model_pic = 'model/sysv2_u1200e100d3dp5.png'  # 模型图保存地址
log = './log/sysv2_u1200e100d3dp5.log'

# data path
x_train = np.loadtxt('mj_data/non_king_processed/X_train_non_king_1.txt', delimiter=' ', dtype=np.float16)
x_test = np.loadtxt('mj_data/non_king_processed/X_test_non_king_1.txt', delimiter=' ', dtype=np.int32)
y_train = np.loadtxt('mj_data/non_king_processed/Y_train_non_king_1.txt', delimiter=' ', dtype=np.float16)
y_test = np.loadtxt('mj_data/non_king_processed/Y_test_non_king_1.txt', delimiter=' ', dtype=np.int32)

y_train = np_utils.to_categorical(y_train, num_classes=34)
y_test = np_utils.to_categorical(y_test, num_classes=34)

# model define
model = Sequential()
model.add(Dense(units, input_dim=input_features, activation='relu'))
model.add(Dense(units, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units, activation='relu'))
model.add(Dropout(0.5))
# model.add(Dense(output_features))
model.add(Dense(output_features, input_dim=output_features))
# model.add(Dense(output_features, input_dim=output_features, kernel_regularizer=regularizers.l2(0.001),
#                 activity_regularizer=regularizers.l1(0.001)))
model.add(Activation('softmax'))

adam = Adam(lr=1e-4)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
print(model)

# 创建一个实例history
history = LossHistory()
plot_model(model, to_file=model_pic)

# 载入模型
# model = load_model(model_restore)
# print('Model Restore!')

# 开始训练和测试
print('Training ------------')
hist_log = model.fit(x_train, y_train, batch_size=batch, epochs=num_epoch, validation_data=(x_test, y_test),
                     callbacks=[history])
with open(log, 'w') as f:
    f.write(str(hist_log.history))

# 评估模型
print('\nTesting ------------')
loss, accuracy = model.evaluate(x_test, y_test)
print('\ntest loss', loss)
print('accuracy', accuracy)

# 保存模型
model.save(model_save_path)

# 绘制acc-loss曲线
history.loss_plot('epoch')

# 运行时间
endtime = datetime.datetime.now()
print('usetime | ', endtime - starttime)
