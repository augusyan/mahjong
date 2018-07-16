# -*- coding:utf-8 -*-
"""
@author : Augus Yan
@file : model_res34_k2tf.py
@time : 2018/7/12 16:09
@function : 
"""
import tensorflow as tf
from keras import backend as K
import keras
import time
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
from keras import regularizers
from keras.models import Sequential
from keras.objectives import categorical_crossentropy

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
        plt.savefig('loss/loss_sysv2_res50.png')
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


def Conv_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x


def model_test(model_path,X):
    # 载入模型
    model = load_model(model_path)   # 载入模型
    print('Model Restore!')
    print(model.predict_classes(X, batch_size=1, verbose=0))

# time record
t0 = time.clock()

# global define
batch = 32  # 批次大小
input_features = 361  # 数据的维数
output_features = 34  # 输出类别
units = 1024  # fc神经元数
num_epoch = 50  # 训练轮数
model_save_path = 'model/sys_res50.h5'  # 模型保存地址
model_pic = 'model/sys_res50.png'  # 绘图保存地址
res_block = 7   # 残差块
log = './log/sys_res50.log'  #log 地址

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

data_size = len(x_train)
num_batches_per_epoch = int(data_size / batch)
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

def _build_residual_block( x, index):
    # mc = self.config.model
    in_x = x
    res_name = "res" + str(index)
    x = Conv2D(filters=256, kernel_size=3, padding="same",
               data_format="channels_first", use_bias=False, kernel_regularizer=l2(0.001),
               name=res_name + "_conv1-" + str(3) + "-" + str(3))(x)
    x = BatchNormalization(axis=1, name=res_name + "_batchnorm1")(x)
    x = Activation("relu", name=res_name + "_relu1")(x)
    x = Conv2D(filters=256, kernel_size=3, padding="same",
               data_format="channels_first", use_bias=False, kernel_regularizer=l2(0.001),
               name=res_name + "_conv2-" + str(3) + "-" + str(3))(x)
    x = BatchNormalization(axis=1, name="res" + str(index) + "_batchnorm2")(x)
    x = Add(name=res_name + "_add")([in_x, x])
    x = Activation("relu", name=res_name + "_relu2")(x)
    return x

with tf.name_scope('input'):
    # this place holder is the same with input layer in keras
    x = tf.placeholder(tf.float32, shape=(None, 1, 19, 19))
    y_ = tf.placeholder(tf.float32, shape=(None, 34))

# in_x = x = Input((1, 19, 19))
# (batch, channels, height, width)


x = Conv2D(filters=256, kernel_size=5, padding="same",
               data_format="channels_first", use_bias=False, kernel_regularizer=l2(0.001),
               name="input_conv-" + str(5) + "-" + str(256))(x)
x = BatchNormalization(axis=1, name="input_batchnorm")(x)
x = Activation("relu", name="input_relu")(x)

for i in range(res_block):
    x = _build_residual_block(x, i + 1)
res_out = x

x = AveragePooling2D(pool_size=(7, 7))(x)

x = Flatten()(x)
preds = Dense(34, activation='softmax')(x)

loss = tf.reduce_mean(categorical_crossentropy(y_, preds))
optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)
# Metric
correct_prediction = tf.equal(tf.argmax(preds, 1), tf.cast(y_, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# model = Model(inputs=in_x, outputs=x)
# model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# model.summary()

saver = tf.train.Saver()

sess = tf.Session()
K.set_session(sess)

# 合并到Summary中
# merged = tf.summary.merge_all()
# writer = tf.summary.FileWriter("graph/", sess.graph)
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
for epoch in range(num_epoch):
    # print('epoch:', epoch)
    shuffle_indices = np.random.permutation(np.arange(data_size))
    x_data = x_train[shuffle_indices]
    y_data = x_test[shuffle_indices]
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch
        end_index = min((batch_num + 1) * batch, data_size)
        x_data = x_data[start_index:end_index]
        y_data = y_data[start_index:end_index] # .astype(np.int64)
        feed = {x: x_data, y_: y_data}
        optimizer.run(feed_dict=feed)
    train_accuracy = accuracy.eval(feed_dict=feed)
    # result = sess.run(merged, feed_dict=feed)  # merged也是需要run的
    # writer.add_summary(result, epoch)  # result是summary类型的，需要放入writer中，i步数（x轴）
    if epoch % 20 == 0:
        saver_path = saver.save(sess, 'save/' + model_save_path + '_%d.ckpt' % epoch)
    print("epoch %04d | training_accuracy %.6f" % (epoch, train_accuracy))
print('-----Testing-----')

feed = {x: x_test, y_: y_test}
test_num = x_test.shape[0]
test_accuracy = accuracy.eval(feed_dict=feed)
print("test_number %04d | testing_accuracy %.9f" % (test_num, test_accuracy))
print('-+---------------------------+-')
print("Model saved in file:", saver_path)
sess.close()

print('usetime | ', t0 - time.clock())
