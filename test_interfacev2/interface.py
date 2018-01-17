# -*- coding: utf-8 -*-
# @Time    : 17-12-28 上午9:26
# @Author  : Yan
# @Site    : 
# @File    : interface.py
# @Software: PyCharm Community Edition
# @Function: 丢弃了原来的门清模型，引入综合模型


from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import os, json, copy
import sys
import feature_extract_v7 as feature_extract


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
            W = tf.Variable(tf.truncated_normal([n_features, n_labels], stddev=(2.0 / n_features)),
                            name='weights%d' % n_layer)  # Weight中都是随机变量
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
            tf.summary.histogram(layer_name + "/weights", W)  # 可视化观看变量
        with tf.name_scope('biases%d' % n_layer):
            b = tf.Variable(tf.zeros([n_labels]), name='biases%d' % n_layer)  # biases推荐初始值不为0
            tf.summary.histogram(layer_name + "/biases", b)  # 可视化观看变量
        with tf.name_scope('hidden%d' % n_layer):
            h = tf.add(tf.matmul(inputs, W), b, name='hidden%d' % n_layer)  # inputs*Weight+biases
            tf.summary.histogram(layer_name + "/hidden", h)  # 可视化观看变量
        if activation is None:
            outputs = h
        elif activation == 'relu':
            outputs = tf.nn.relu(h)
            outputs = tf.nn.dropout(outputs, keep_prob=1)
        tf.summary.histogram(layer_name + "/outputs", outputs)  # 可视化观看变量
    return outputs


# 模型调用
def model_path(ismyturn, isking):
    """
    Return model path
    :param ismyturn: is my turn
    :param isking: is a king card in my hands
    :return: nn_units,input_features,output_features,model_restore_path
    """
    # 判断是不是自己的回合
    if ismyturn:
        # 判断手牌中是否有宝牌
        if isking:
            nn_units = 1200
            input_features = 296
            output_features = 34
            model_restore_path = "save/20171227_king_v0_layer2_d10_c9_100.ckpt"
        else:
            nn_units = 1200
            input_features = 292
            output_features = 34
            model_restore_path = "save/20171227_non_king_v0_layer2_d10_c9_100.ckpt"
    else:
        if isking:
            nn_units = 1000
            input_features = 213
            output_features = 8
            model_restore_path = "save/20171225_op_king2_v0_layer2_d10_continue_100.ckpt"
        else:
            nn_units = 1000
            input_features = 210
            output_features = 8
            model_restore_path = "save/20171225_non_king2_v0_layer2_d10_continue_100.ckpt"

    return nn_units, input_features, output_features, model_restore_path


# 模型选取和训练
def model_choose(ismyturn, isking, list, hand_cards):
    """
    Return model decision
    :param ismyturn: is my turn
    :param isking: is a king card in my hands
    :param list: hand cards and suits cards
    :param hand_cards: hand cards and suits cards
    :return: result1
    """
    # init and restart graph
    tf.reset_default_graph()

    # data path
    testData = np.array(list)

    # global config
    learning_rate = 0.0001
    beta = 0.001
    keep_prob = 1
    test_epochs = 1

    nn_units, input_features, output_features, model_restore_path = model_path(ismyturn, isking)
    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.float32, shape=[None, input_features], name='x')
        y_ = tf.placeholder(tf.int32, shape=[None], name='y_')

    # weights & bias for nn layers correct_prediction
    layer1 = add_layer(x, input_features, nn_units, 1, 'relu')
    layer2 = add_layer(layer1, nn_units, nn_units, 2, 'relu')
    output = add_layer(layer2, nn_units, output_features, 3)

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
    tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    # model running
    print('---testing---')
    saver.restore(sess, model_restore_path)
    print('model restore !')
    # data input
    x_data = np.array(testData[0:-1])
    sign = 1
    test_data = x_data
    # print(test_label.shape)
    ret = sess.run(L1, feed_dict={x: test_data.reshape(1, input_features)})

    # 判断是否是自己的回合，是则决策出牌，不是则决策吃碰杠
    if ismyturn:
        best = ret.argmax() + 1
        # print(handCards)
        # print('best:%d' %best)
        while sign:
            if (translate2(ret.argmax() + 1)) not in hand_cards:
                # print('ret.argmax %d' %ret.argmax())
                ret[0][ret.argmax()] = 0
            else:
                sign = 0
        print('---')
        print(ret[0][ret.argmax()])
        print('---')
        print('our decision:%d | ' % (translate(ret.argmax() + 1)) + 'best decision:%d | ' % (translate(best)))
        result = (translate(ret.argmax() + 1))
    else:
        decision = ret.argmax()
        op_out_table = {0: '不操作', 1: '吃左', 2: '吃中', 3: '吃右', 4: '碰', 5: '明杠', 6: '暗杠', 7: '补杠'}
        print('our decision:%s | ' % op_out_table[decision] + 'best decision:%s | ' % op_out_table[decision])
        result = decision
    sess.close()
    return result


def translate(i):
    if i >= 1 and i <= 9:
        return i
    elif i >= 10 and i <= 18:
        return i + 1
    elif i >= 19 and i <= 27:
        return i + 2
    elif i >= 28 and i <= 34:
        return i + 3
    else:
        print('Error !')


def translate2(i):  # 转换十进制到cards
    if i >= 10 and i <= 18:
        i = i + 7
    elif i >= 19 and i <= 27:
        i = i + 14
    elif i >= 28 and i <= 34:
        i = i + 21
    return i


dict = {}
# 全局变量
handCards = []  # 手牌
king_card = 0x00  # 宝牌
op_card = 0x00  # 操作牌
feature = []  # 特征值
feature_noking = []  # 无宝特征
feature_king = []  # 有宝特征
fei_king = 0

inits = input("请输入初始手牌,每张牌用，隔开:\n")
print(type(inits))
str_array = inits.split(',')
init_cards = []
for i in str_array:
    temp = int(i[0]) * 16 + int(i[1])
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
handCards = init_cards  # 更新手牌

king = input('请输入本局的宝牌\n')
flag2 = True
while flag2:
    f = input('输入确认y/n\n')
    if f == 'y':
        flag2 = False
    elif f == 'n':
        king = input('请输入本局的宝牌\n')
king_card = int(king[0]) * 16 + int(king[1])
# 更新king
print("请输入抓牌出牌的牌，抓牌用g开头，出牌用d开头，例如抓1万，g01,出1万，d01")
while 1:
    inputs = input("请输入操作+牌\n")

    if inputs[0] == "G" or inputs[0] == "g":
        op_card = int(inputs[1]) * 16 + int(inputs[2])
        handCards.append(op_card)
        handCards.sort()  # 更新手牌和op_card
        # 调用特征函数写
        print(handCards)
        if king_card not in handCards:
            feature_noking = feature_extract.calculate1(handCards)
            feature_noking.append(35)
            temp = model_choose(isking=False, list=feature_noking, hand_cards=handCards)
            op_card = (int(temp / 10)) * 16 + (temp % 10)
            handCards.remove(op_card)
            print(handCards)
            if op_card == king_card:
                fei_king = fei_king + 1


        else:
            handCards0 = copy.deepcopy(handCards)  # 得到去掉king_card后的手牌，为以后使用
            king_num = 0

            while 1:  # 手牌中的king_card 计数
                if king_card in handCards0:
                    handCards0.remove(king_card)
                    king_num = king_num + 1
                else:
                    break
            feature_king = feature_extract.calculate2(handCards, king_card, king_num, fei_king, 0)
            feature_king.append(35)
            temp = model_choose(isking=True, list=feature_king, hand_cards=handCards)
            op_card = (int(temp / 10)) * 16 + (temp % 10)
            handCards.remove(op_card)
            print(handCards)
            if op_card == king_card:
                fei_king = fei_king + 1

    # elif inputs[0] == "d" or inputs[0] == "D":
    #    op_card = int(inputs[1]) * 16 + int(inputs[2])
    #    handCards.remove(op_card)
    #    print(handCards)
    #     if op_card == king_card:
    #        fei_king = fei_king + 1


    elif inputs[0] == "A" or inputs[0] == "a":
        break
    else:
        print("请重新输入\n")
        continue


        # 01,01,01,02,02,02,03,03,03,04,04,04,05
        # 01,04,07,11,14,18,21,24,29,31,31,34,37
        # 33
        # 1, 9, 9, 17, 17, 21, 25, 25, 41, 41, 49, 49, 51, 52
        # 03,03,04,07,19,21,26,28,28,31,34,36,36
