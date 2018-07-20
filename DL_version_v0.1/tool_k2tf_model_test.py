# -*- coding: utf-8 -*-
# @Time    : 18-7-20 上午11:09
# @Author  : Yan
# @Site    : 
# @File    : tool_k2tf_model_test.py
# @Software: PyCharm Community Edition
# @Function: 
# @update:

import tensorflow as tf
import numpy as np

# data path
x_test = np.loadtxt('mj_data/data_201804/king_20180528/output_X_test_pic.txt', delimiter=' ', dtype=np.int32)
y_test = np.loadtxt('mj_data/data_201804/king_20180528/output_Y_test.txt', delimiter=' ', dtype=np.int32)

# f1 = np.reshape(x_test[100000],(-1, 253))
# x_test = x_test.reshape(-1, 253)

f1 = np.reshape(x_test[100000],(-1,1,19,19))
f2 = y_test[100000]
print(np.shape(f1))
print(f2)

export_dir = './model/20180528/k2tf_model/sys_king_res34_e50d5/'

with tf.Session(graph=tf.Graph()) as sess:
    meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
    signature = meta_graph_def.signature_def

    x1_tensor_name = signature['predict_webshell_php'].inputs['main_input'].name
    # x2_tensor_name = signature['predict_webshell_php'].inputs['feature_input'].name
    y_tensor_name = signature['predict_webshell_php'].outputs[
        tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES].name

    x1 = sess.graph.get_tensor_by_name(x1_tensor_name)
    # x2 = sess.graph.get_tensor_by_name(x2_tensor_name)
    y = sess.graph.get_tensor_by_name(y_tensor_name)
    print(x1, y)

    print(sess.run(y, feed_dict={x1: f1}))  # 预测值
