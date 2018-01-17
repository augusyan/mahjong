# -*- coding: utf-8 -*-
# @Time    : 18-1-9 下午5:21
# @Author  : Yan
# @Site    : 
# @File    : model.py
# @Software: PyCharm Community Edition
# @Function:
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np


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
            model_restore_path = "save/20171227_sys_king_v0_layer2_d10_c9_100.ckpt"
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
    :return: result
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
    print(test_data.shape)
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
