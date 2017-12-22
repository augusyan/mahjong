# -*-coding:utf-8 -*-
# __author__='Yan'
# function: processing op kings model,features = 213  20171221

from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import os
import datetime
from sklearn import cross_validation

# choose gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# add layers
def add_layer(inputs, n_features, n_labels, n_layer, activation=None):
    """
    Return TensorFlow weights
    :param n_features: Number of features
    :param n_labels: Number of labels
    :return: TensorFlow hidden layer
    """
    layer_name = "layer%d" % n_layer
    regularizer = layers.l2_regularizer(beta)
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
            outputs = tf.nn.dropout(outputs, keep_prob=keep_prob)
        tf.summary.histogram(layer_name + "/outputs", outputs)  # 可视化观看变量
    return outputs


# transform to one-hot coding
def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


# data to batch
def batch_iter(sourceData, batch_size, num_epochs, shuffle=True):
    # data = np.array(sourceData)  # 将sourceData转换为array存储
    data_size = len(sourceData)
    num_batches_per_epoch = int(len(sourceData) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = sourceData[shuffle_indices]
        else:
            shuffled_data = sourceData

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            yield shuffled_data[start_index:end_index]



# time record
starttime = datetime.datetime.now()

# data path
trainData_tmp = np.loadtxt('data/op_kings_train.txt', delimiter=' ', dtype=np.float16)
# trainData_tmp = np.loadtxt('data/data223_train.txt', delimiter=' ', dtype=np.float16)
trainData = trainData_tmp
testData_tmp = np.loadtxt('data/op_kings_test.txt', delimiter=' ', dtype=np.float16)
testData = testData_tmp

# show the input data
# print('---input data---')
# print(trainData)
# print(trainData.shape)
# x_data = np.array(trainData[:, 0:-1])
# y_data = np.array(trainData[:, -1]).astype(np.int32)  # .astype(np.int64)
# data = batch_iter(trainData,50,1)

"""
# queue
q = tf.FIFOQueue(capacity=5, dtypes=tf.float32)  # enqueue 5 batches
# We use the "enqueue" operation so 1 element of the queue is the full batch
enqueue_op = q.enqueue(x_data)
numberOfThreads = 1
qr = tf.train.QueueRunner(q, [enqueue_op] * numberOfThreads)
tf.train.add_queue_runner(qr)
x = q.dequeue()  # It replaces our input placeholder
print(x)


batch_size = 50
mini_after_dequeue = 1000
capacity = mini_after_dequeue+3*batch_size

example_batch, label_batch = tf.train.shuffle_batch(
    tensors=[x_data, y_data],
    batch_size=batch_size,
    capacity=capacity,
    min_after_dequeue=mini_after_dequeue
)

#batch_size = 4
#mini_after_dequeue = 100
#capacity = mini_after_dequeue+3*batch_size

#example_batch,label_batch = tf.train.batch([image,label],batch_size = batch_size,capacity=capacity)
"""

# global config
learning_rate = 0.0001
beta = 0.001
keep_prob = 1
num_epochs = 101
test_epochs = 30
batch_size = 50
input_features = 212
output_features = 8

data_size = len(trainData)
num_batches_per_epoch = int(data_size / batch_size)
model_save_name = '20171221_op_king_v0_layer2_d10'
model_restore_path = "save/20171211_king_v0_layer2_d10_100.ckpt"

# input placeholder
with tf.name_scope('inputs'):
    x = tf.placeholder(tf.float32, shape=[None, input_features], name='x')
    y_ = tf.placeholder(tf.int32, shape=[None], name='y_')

# weights & bias for nn layerscorrect_prediction
layer1 = add_layer(x, input_features, 1000, 1, 'relu')
layer2 = add_layer(layer1, 1000, 1000, 2, 'relu')
# layer3 = add_layer(layer2, 1000, 1000, 3, 'relu')
# output = add_layer(layer3, 1000, 35, 4)
output = add_layer(layer2, 1000, output_features, 3)

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
print('---training---')


def train_test_model(train=True, show=False, continue_train = False):
    if train:
        with tf.Session() as sess:
            # 合并到Summary中
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter("graph/", sess.graph)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            for epoch in range(num_epochs):
                # print('epoch:', epoch)
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = trainData[shuffle_indices]
                for batch_num in range(num_batches_per_epoch):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size)
                    data_batch = shuffled_data[start_index:end_index]
                    x_data = np.array(data_batch[:, 0:-1])
                    y_data = np.array(data_batch[:, -1]).astype(np.int32)  # .astype(np.int64)
                    feed = {x: x_data, y_: y_data}
                    optimizer.run(feed_dict=feed)
                train_accuracy = accuracy.eval(feed_dict=feed)
                result = sess.run(merged, feed_dict=feed)  # merged也是需要run的
                writer.add_summary(result, epoch)  # result是summary类型的，需要放入writer中，i步数（x轴）
                if epoch % 20 == 0:
                    saver_path = saver.save(sess, 'save/' + model_save_name + '_%d.ckpt' % epoch)
                print("epoch %04d | training_accuracy %.6f" % (epoch, train_accuracy))
            print('-----Testing-----')
            x_data = np.array(testData[:, 0:-1])
            y_data = np.array(testData[:, -1]).astype(np.int32)  # .astype(np.int64)
            feed = {x: x_data, y_: y_data}
            test_num = x_data.shape[0]
            test_accuracy = accuracy.eval(feed_dict=feed)
            print("test_number %04d | testing_accuracy %.9f" % (test_num, test_accuracy))
        print('-+---------------------------+-')
        print("Model saved in file:", saver_path)

    else:
        with tf.Session() as sess:
            saver.restore(sess, model_restore_path)
            print('model restore !')
            print('-----Testing-----')
            x_data = np.array(testData[:, 0:-1])
            y_data = np.array(testData[:, -1]).astype(np.int32)  # .astype(np.int64)
            feed = {x: x_data, y_: y_data}
            test_num = x_data.shape[0]
            test_accuracy = accuracy.eval(feed_dict=feed)
            print("test_samples %04d | testing_accuracy %.9f" % (test_num, test_accuracy))

    if show:
        with tf.Session() as sess:
            saver.restore(sess, model_restore_path)
            print('model restore !')
            # data input
            x_data = np.array(testData[:, 0:-1])
            y_data = np.array(testData[:, -1]).astype(np.int32)  # .astype(np.int64)
            for step_test in range(test_epochs):
                test_data = x_data[step_test]
                test_label = y_data[step_test]
                # print(test_label.shape)
                ret = sess.run(L1, feed_dict={x: test_data.reshape(1, input_features)})
                print('---')
                print(ret)
                print('---')
                print('hyperthesis:%d | ' % (ret.argmax())+'true Y:%d' % (test_label))

    if continue_train:
        with tf.Session() as sess:
            saver.restore(sess, model_restore_path)
            print('model restore !')
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter("graph/", sess.graph)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            for epoch in range(num_epochs):
                # print('epoch:', epoch)
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = trainData[shuffle_indices]
                for batch_num in range(num_batches_per_epoch):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size)
                    data_batch = shuffled_data[start_index:end_index]
                    x_data = np.array(data_batch[:, 0:-1])
                    y_data = np.array(data_batch[:, -1]).astype(np.int32)  # .astype(np.int64)
                    feed = {x: x_data, y_: y_data}
                    optimizer.run(feed_dict=feed)
                train_accuracy = accuracy.eval(feed_dict=feed)
                result = sess.run(merged, feed_dict=feed)  # merged也是需要run的
                writer.add_summary(result, epoch)  # result是summary类型的，需要放入writer中，i步数（x轴）
                if epoch % 20 == 0:
                    saver_path = saver.save(sess, 'save/' + model_save_name + '_continue_%d.ckpt' % epoch)
                print("epoch %04d | training_accuracy %.6f" % (epoch, train_accuracy))
            print('-----Testing-----')
            x_data = np.array(testData[:, 0:-1])
            y_data = np.array(testData[:, -1]).astype(np.int32)  # .astype(np.int64)
            feed = {x: x_data, y_: y_data}
            test_num = x_data.shape[0]
            test_accuracy = accuracy.eval(feed_dict=feed)
            print("test_number %04d | testing_accuracy %.9f" % (test_num, test_accuracy))
        print('-+---------------------------+-')
        print("Model saved in file:", saver_path)



if __name__ == '__main__':
    train_test_model(train=False,show=True)
    endtime = datetime.datetime.now()
    print(endtime - starttime)