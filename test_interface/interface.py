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
import os
import datetime
from sklearn import cross_validation
import os, json, copy
import sys
import write_feature


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
    W = tf.Variable(tf.truncated_normal([n_features, n_labels], stddev=(2.0 / n_features)),name='weights%d' % n_layer)  # Weight中都是随机变量
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
    b = tf.Variable(tf.zeros([n_labels]),name='biases%d' % n_layer)  # biases推荐初始值不为0
    h = tf.add(tf.matmul(inputs, W), b,name='hidden%d' % n_layer)  # inputs*Weight+biases
    if activation is None:
        target = h
    elif activation == 'relu':
        target = tf.nn.relu(h)
        target = tf.nn.dropout(target, keep_prob=1.0)
    return target


# 模型选取和训练
def model_choose(isking, list):
    # time record
    starttime = datetime.datetime.now()

    # data path
    testData_tmp = np.loadtxt('data/kings_test_3.txt', delimiter=' ', dtype=np.float16)
    testData = testData_tmp

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

    # input placeholder
    x = tf.placeholder(tf.float32, shape=[None, input_features], name='x')
    y_ = tf.placeholder(tf.int32, shape=[None], name='y_')

    # weights & bias for nn layerscorrect_prediction
    layer1 = add_layer(x, input_features, 1000, 1, 'relu')
    layer2 = add_layer(layer1, 1000, 1000, 2, 'relu')
    # layer3 = add_layer(layer2, 1000, 1000, 3, 'relu')
    # output = add_layer(layer3, 1000, 35, 4)
    output = add_layer(layer2, 1000, 34, 3)

    # Regularization L2
    L1 = tf.nn.softmax(output)

    regularizer = tf.contrib.layers.l2_regularizer(beta)
    reg_term = tf.contrib.layers.apply_regularization(regularizer)

    # optimizer paramaters
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=output) + reg_term
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(L1, 1), tf.cast(y_, tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # save model
    saver = tf.train.Saver()

    # model running
    print('---training---')
    with tf.Session() as sess:
        saver.restore(sess, model_restore_path)
        print('model restore !')
        # data input
        x_data = np.array(testData[:, 0:-1])
        y_data = np.array(testData[:, -1]).astype(np.int32)  # .astype(np.int64)
        for step_test in range(test_epochs):
            sign = 1
            test_data = x_data[step_test]
            test_label = y_data[step_test]
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
            print('our decision:%d | ' % (ret.argmax()) + 'best decision:%d | ' % (best) + 'Y:' % (test_label))


dict = {}

inits = input("请输入初始手牌,每张牌用，隔开:\n")
str_array = inits.split(',')
init_cards = []
for i in str_array:
    init_cards.append(int(i))
# 输入正确性判断
while len(init_cards) != 13 and len(init_cards) != 14:
    print('输入有误，请重新输入\n')
    inits = input("请输入初始手牌:\n")
    str_array = inits.split(',')
    init_cards = []
    for i in str_array:
        init_cards.append(int(i))

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
        init_cards[int(index)] = int(card)

dict["init_cards"] = init_cards
actions = []

'''
{"combine_cards": [],
 "seat_id": 2, 
 "op_card": 255,
  "action_type": "A",
   "out_seat_id": 255}
    
'''

king_card = input('请输入本局的宝牌\n')
flag2 = True
while flag2:
    f = input('输入确认y/n\n')
    if f == 'y':
        flag2 = False
    elif f == 'n':
        king_card = input('请输入本局的宝牌\n')
king_cards = [int(king_card)]
dict['king_cards'] = king_cards

print("请输入抓牌出牌的牌，抓牌用g开头，出牌用d开头，例如抓1万，g01,出1万，d01")
while 1:
    inputs = input("牌\n")
    action = {
        "combine_cards": [],
        "seat_id": 0,
        "action_type": "g",
        "out_seat_id": 255
    }
    if inputs[0] == "G" or inputs[0] == "g":
        action["combine_cards"] = []
        action["seat_id"] = 0
        action["op_card"] = int(inputs[1]) * 16 + int(inputs[2])
        action["action_type"] = "G"
        action["out_seat_id"] = 255
        actions.append(action)
        dict["actions"] = actions
        with open("./input", 'w') as json_file:
            json.dump(dict, json_file)
        fo = open("./output.txt", "w")  # 写的文件
        fo.truncate()  # 清空output
        with open("./input", 'r') as json_file:  # 打开文件
            data = json.load(json_file)
            write_feature.write(fo, data)
        fo.close()

    elif inputs[0] == "d" or inputs[0] == "D":
        action["combine_cards"] = []
        action["seat_id"] = 0
        action["op_card"] = int(inputs[1]) * 16 + int(inputs[2])
        action["action_type"] = "d"
        action["out_seat_id"] = 255
        actions.append(action)
        dict["actions"] = actions
        with open("./input", 'w') as json_file:
            json.dump(dict, json_file)

    elif inputs[0] == "A" or inputs[0] == "a":
        action["combine_cards"] = []
        action["seat_id"] = 0
        action["op_card"] = 255
        action["action_type"] = "A"
        action["out_seat_id"] = 255
        actions.append(action)
        dict["actions"] = actions
        with open("./input", 'w') as json_file:
            json.dump(dict, json_file)
        break
    else:
        print("请重新输入\n")
        continue
