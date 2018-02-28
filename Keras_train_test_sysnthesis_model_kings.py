# -*- coding: utf-8 -*-
# @Time    : 18-2-27 下午4:39
# @Author  : Yan
# @Site    : 
# @File    : sys_model_test0.py
# @Software: PyCharm Community Edition
# @Function: keras achieve mahjong model
# @update:

from __future__ import division
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.datasets import cifar100


batch_size = 50
input_features = 296
output_features = 34

# data path
trainData_tmp = np.loadtxt('sys_kings_train_5.txt', delimiter=' ', dtype=np.float16)
# trainData_tmp = np.loadtxt('dataset_v11_train_test.txt', delimiter=' ', dtype=np.float16)
testData_tmp = np.loadtxt('sys_kings_test_5.txt', delimiter=' ', dtype=np.float16)
# testData_tmp = np.loadtxt('dataset_v11_mini_test.txt', delimiter=' ', dtype=np.float16)

x_train = np.array(trainData_tmp[:, 0:-1])
# x_train = x_train.reshape(-1, 23)
y_train = np.array(trainData_tmp[:, -1]).astype(np.int32)
y_train = np_utils.to_categorical(y_train, num_classes=34)

x_test = np.array(testData_tmp[:, 0:-1])
# x_train = x_train.reshape(-1, 23)
y_test = np.array(testData_tmp[:, -1]).astype(np.int32)
y_test = np_utils.to_categorical(y_test, num_classes=34)

print(np.shape(x_train))
print(np.shape(y_train))
print(np.shape(x_test))
print(np.shape(y_test))

# model define
model = Sequential()
model.add(Dense(1200, input_dim=input_features, activation='relu'))
model.add(Dense(1200, activation='relu'))
model.add(Dense(output_features))
model.add(Activation('softmax'))

# Another way to define your optimizer
adam = Adam(lr=1e-4)

# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
# print(model)

print('Training ------------')
model.fit(x_train, y_train, epochs=2)
print('\nTesting ------------')
loss, accuracy = model.evaluate(x_test, y_test)
print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

# data_size = len(trainData)
# num_batches_per_epoch = int(data_size / batch_size)
# shuffle_indices = np.random.permutation(np.arange(data_size))
# shuffled_data = trainData[shuffle_indices]
# # Another way to train the model
# for batch_num in range(num_batches_per_epoch):
#     start_index = batch_num * batch_size
#     end_index = min((batch_num + 1) * batch_size, data_size)
#     data_batch = shuffled_data[start_index:end_index]
#     x_data = np.array(data_batch[:, 0:-1])
#     y_data = np.array(data_batch[:, -1]).astype(np.int32)
#     model.fit(x_data, y_data, epochs=1)
#     print('\nTesting ------------')
#     # Evaluate the model with the metrics we defined earlier
#     X_test = np.array(testData[:, 0:-1])
#     y_test = np.array(testData[:, -1]).astype(np.int32)  # .astype(np.int64)
#
#     loss, accuracy = model.evaluate(X_test, y_test)
#
#     print('\ntest loss: ', loss)
#     print('\ntest accuracy: ', accuracy)
