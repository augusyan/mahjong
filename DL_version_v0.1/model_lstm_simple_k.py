# -*- coding: utf-8 -*-
# @Time    : 18-5-4 上午9:02
# @Author  : Yan
# @Site    : 
# @File    : model_lstm_simple_k.py
# @Software: PyCharm Community Edition
# @Function: a MNIST in lstm and achieve by keras
# @update:

import keras
from keras.engine.topology import Input, Layer
from keras.engine.training import Model
from keras.layers import LSTM
from keras.layers import Dense, Activation
from keras.datasets import mnist
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
import numpy as np
from keras.callbacks import TensorBoard

batch = 64  # 批次大小
input_features = 329  # 数据的维数
output_features = 34  # 输出类别
units = 128  # fc神经元数
num_epoch = 100  # 训练轮数
model_save_path = 'model/sys_lstm_simple_e10f329_m.h5'  # 模型保存地址
model_pic = 'model/sys_lstm_simple_e10f329_m.png'  # 绘图保存地址
log = './log/sys_lstm_simple_e10f329_m.log'  # log 地址
model_restore = 'model/sys_lstm_simple_e10f329_m.h5'
# data path
x_train = np.loadtxt('mj_data/non_king_processed/X_train_non_king_1.txt', delimiter=' ', dtype=np.float16)
x_test = np.loadtxt('mj_data/non_king_processed/X_test_non_king_1.txt', delimiter=' ', dtype=np.int32)
y_train = np.loadtxt('mj_data/non_king_processed/Y_train_non_king_1.txt', delimiter=' ', dtype=np.float16)
y_test = np.loadtxt('mj_data/non_king_processed/Y_test_non_king_1.txt', delimiter=' ', dtype=np.int32)

print(np.shape(x_train))
print(np.shape(x_test))
print(np.shape(y_train))
print(np.shape(y_test))


y_train = np_utils.to_categorical(y_train, num_classes=34)
y_test = np_utils.to_categorical(y_test, num_classes=34)

# reshape
x_train = x_train.reshape(-1, 1, input_features)
x_test = x_test.reshape(-1, 1, input_features)

model = Sequential()
model.add(LSTM(units,input_shape=(input_features, 1),
               unroll=False))
# model.add(LSTM(units,input_shape=(input_features, 1),
#                unroll=False))

model.add(Dense(output_features))
model.add(Activation('softmax'))

adam = Adam(lr=1e-4)
model.summary()
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

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
