# -*- coding: utf-8 -*-
"""
Created on Sun Nov 05 17:26:14 2017

@author: ren
"""

from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import os, json, copy
import sys
import feature_extract_v5


# 模型使用，定义网络层
def add_layer(inputs, n_features, n_labels, n_layer, activation=None):
    """
    Return TensorFlow weights
    :param n_features: Number of features
    :param n_labels: Number of labels
    :return: TensorFlow hidden layer
    """
    layer_name = "layer%d" % n_layer
    regularizer = layers.l2_regularizer(0.001)
    # TODO: Return hidden layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights%d' % n_layer):
            W = tf.Variable(tf.truncated_normal([n_features, n_labels], stddev=(2.0 / n_features)),name='weights%d' % n_layer)  # Weight中都是随机变量
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
            tf.summary.histogram(layer_name + "/weights", W)  # 可视化观看变量
        with tf.name_scope('biases%d' % n_layer):
            b = tf.Variable(tf.zeros([n_labels]),name='biases%d' % n_layer)  # biases推荐初始值不为0
            tf.summary.histogram(layer_name + "/biases", b)  # 可视化观看变量
        with tf.name_scope('hidden%d' % n_layer):
            h = tf.add(tf.matmul(inputs, W), b,name='hidden%d' % n_layer)  # inputs*Weight+biases
            tf.summary.histogram(layer_name + "/hidden", h)  # 可视化观看变量
        if activation is None:
            outputs = h
        elif activation == 'relu':
            outputs = tf.nn.relu(h)
            outputs = tf.nn.dropout(outputs, keep_prob=1)
        tf.summary.histogram(layer_name + "/outputs", outputs)  # 可视化观看变量
    return outputs


# 模型选取和训练
def model_choose(isking, list):
    # data path
    # testData_tmp = np.loadtxt('data/kings_test_3.txt', delimiter=' ', dtype=np.float16)
    testData = np.array(list)
    print(type(testData))
    # global config
    learning_rate = 0.0001
    beta = 0.001
    keep_prob = 1
    test_epochs = 1
    if isking:
        input_features = 206
        model_restore_path = "save/20171211_king3_v0_continue_100.ckpt"

    else:
        input_features = 202
        model_restore_path = "save/20171211_non_king3_v0_layer2_continue_100.ckpt"

    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.float32, shape=[None, input_features], name='x')
        y_ = tf.placeholder(tf.int32, shape=[None], name='y_')

    # weights & bias for nn layerscorrect_prediction
    layer1 = add_layer(x, input_features, 1000, 1, 'relu')
    layer2 = add_layer(layer1, 1000, 1000, 2, 'relu')
    # layer3 = add_layer(layer2, 1000, 1000, 3, 'relu')
    # output = add_layer(layer3, 1000, 35, 4)
    output = add_layer(layer2, 1000, 34, 3)

    # tf.add_to_collection(tf.GraphKeys.WEIGHTS, W_2)
    # tf.add_to_collection(tf.GraphKeys.WEIGHTS, W_3)
    # reg_term = tf.contrib.layers.apply_regularization(regularizer)
    # loss = (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_,logits=z_3)) + reg_term)

    # Regularization L2
    L1 = tf.nn.softmax(output)

    regularizer = tf.contrib.layers.l2_regularizer(beta)
    reg_term = tf.contrib.layers.apply_regularization(regularizer)

    # optimizer paramaters
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=output) + reg_term
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(L1, 1), tf.cast(y_, tf.int64))
    # correct_prediction = tf.equal(tf.argmax(L1,1), tf.argmax(y_,1))

    with tf.name_scope('loss'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.summary.scalar('loss', accuracy)
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # save model
    saver = tf.train.Saver()

    # model running
    print('---training---')
    with tf.Session() as sess:
        saver.restore(sess, model_restore_path)
        print('model restore !')
        # data input
        x_data = np.array(testData[0:-1])
        # y_data = np.array(testData[-1]).astype(np.int32)  # .astype(np.int64)
        sign = 1
        test_data = x_data
        # print(test_label.shape)
        ret = sess.run(L1, feed_dict={x: test_data.reshape(1, input_features)})
        best = ret.argmax()
        while sign:
            if ret.argmax() not in list:
                ret[0][ret.argmax()] = 0
            else:
                sign = 0
        print('---')
        print(ret[0][ret.argmax()])
        print('---')
        print('our decision:%d | ' % (ret.argmax()) + 'best decision:%d | ' % (best))


sess = tf.Session()
dict = {}
#全局变量
handCards=[] #手牌
king_card = 0x00 #宝牌
op_card=0x00 #操作牌
feature=[] #特征值
feature_noking=[] #无宝特征
feature_king = []#有宝特征

inits = input("请输入初始手牌,每张牌用，隔开:\n")
print(type(inits))
str_array = inits.split(',')
init_cards = []
for i in str_array:
    temp=int(i[0])*16+int(i[1])
    init_cards.append(temp)

# 输入正确性判断
while len(init_cards) != 13 and len(init_cards) != 14:
    print('输入有误，请重新输入\n')
    inits = input("请输入初始手牌:\n")
    str_array = inits.split(',')
    init_cards = []
    for i in str_array:
        temp = int(i[0]) * 16 + int(i[1])
        init_cards.append(temp)

# 输入修改
flag = True
while flag:
    print('输入确认：正确请输入y，需要修改请输入n\n')
    f = input("y/n")
    if f == 'y':
        flag = False
    elif f == 'n':
        index = input('请输入需要修改的是第几张牌\n')
        while int(index) >= len(init_cards) or int(index) < 0:
            print('输入错误,请重新输入\n')
            index = input('请输入需要修改的是第几张牌\n')
        card = input('请输入需要修改的牌值\n')
        temp = int(card[0]) * 16 + int(card[1])
        init_cards[int(index)] = int(temp)
handCards=init_cards #更新手牌





king = input('请输入本局的宝牌\n')
flag2 = True
while flag2:
    f = input('输入确认y/n\n')
    if f == 'y':
        flag2 = False
    elif f == 'n':
        king = input('请输入本局的宝牌\n')
king_card = int(king[0]) * 16 + int(king[1])
  #更新king
print("请输入抓牌出牌的牌，抓牌用g开头，出牌用d开头，例如抓1万，g01,出1万，d01")
while 1:
    inputs = input("请输入操作+牌\n")

    if inputs[0] == "G" or inputs[0] == "g":
        op_card=int(inputs[1]) * 16 + int(inputs[2])
        handCards.append(op_card)
        handCards.sort()  #更新手牌和op_card
        #调用特征函数写
        print (handCards)
        if king_card not in handCards:
            feature_noking=feature_extract_v5.calculate1(handCards)
            feature_noking.append(35)
            model_choose(isking=False,list=feature_noking)
        else:
            handCards0 = copy.deepcopy(handCards)  # 得到去掉king_card后的手牌，为以后使用
            king_num=0

            while 1:  # 手牌中的king_card 计数
                if king_card in handCards0:
                    handCards0.remove(king_card)
                    king_num = king_num + 1
                else:
                    break
            feature_king=feature_extract_v5.calculate2(handCards,king_card,king_num,fei_king,0)
            feature_king.append(35)
            model_choose(isking=True, list=feature_king)


    elif inputs[0] == "d" or inputs[0] == "D":
        op_card = int(inputs[1]) * 16 + int(inputs[2])
        handCards.remove(op_card)
        if op_card == king_card:
            fei_king=fei_king+1


    elif inputs[0] == "A" or inputs[0] == "a":
        break
    else:
        print("请重新输入\n")
        continue


# 01,01,01,02,02,02,03,03,03,04,04,04,05
# 33
