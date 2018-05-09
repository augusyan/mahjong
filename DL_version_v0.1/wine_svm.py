# -*- coding:utf-8 -*-
"""
@author : Augus Yan
@file : wine_svm.py
@time : 2018/3/21 21:04
@function : 
"""
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils

# 导入数据集
# iris = datasets.load_wine()
# X = iris.data[:, :2]  # 只取前两维特征,画图方便
# y = iris.target
x_train = np.loadtxt('mj_data/non_king_processed/X_train_non_king_1.txt', delimiter=' ', dtype=np.float16)
x_test = np.loadtxt('mj_data/non_king_processed/X_test_non_king_1.txt', delimiter=' ', dtype=np.int32)
y_train = np.loadtxt('mj_data/non_king_processed/Y_train_non_king_1.txt', delimiter=' ', dtype=np.float16)
y_test = np.loadtxt('mj_data/non_king_processed/Y_test_non_king_1.txt', delimiter=' ', dtype=np.int32)

# y_train = np_utils.to_categorical(y_train, num_classes=34)
# y_test = np_utils.to_categorical(y_test, num_classes=34)

# x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1, train_size=0.6)
h = .02  # 网格中的步长

#C是惩罚因子
#kernel核方法，常用的核方法有：‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
print(y_train.ravel())
clf = svm.SVC(C=1.0, kernel='rbf', gamma=20, degree=3,decision_function_shape='ovr')
starttime = datetime.datetime.now()
clf.fit(x_train, y_train.ravel())
traintime = datetime.datetime.now()
t_train = traintime - starttime
print('Train acc', clf.score(x_train, y_train))  # 精度
y_hat = clf.predict(x_train)
accuracy_score(y_hat, y_train, '训练集')
print('Test acc: ', clf.score(x_test, y_test))
y_hat = clf.predict(x_test)
accuracy_score(y_hat, y_test, '测试集')
testtime = datetime.datetime.now()
t_test = testtime - starttime
print('SV: ', clf.n_support_)
print('train cost: ', t_train, 'test time:  ', t_test)
