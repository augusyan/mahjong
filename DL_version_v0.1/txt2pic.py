# -*- coding: utf-8 -*-
# @Time    : 18-3-1 下午3:17
# @Author  : Yan
# @Site    : 
# @File    : txt2pic.py
# @Software: PyCharm Community Edition
# @Function: for my big attempt,make Mahjong data to pic and use for CNN train
# @update:
import os
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from PIL import Image

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.model_selection import ShuffleSplit

# global setting
pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=np.nan)


def read_file(path):
    """
    从txt/csv格式读取文件
    :param path: Where U put data in the dir
    :return: dft a panda table
    """
    # dfc = pd.read_csv(path)
    # dft = np.loadtxt(path)
    dft = pd.read_table(path, header=None, delim_whitespace=True)
    distrbution = dft.describe()
    print(distrbution)
    return dft


def delete_repeat(filename, line=33):
    """
    粗比的删除重复矛盾的行
    :param filename: Where U put data in the dir
    :param line: Where U put data in the dir
    :return: dft_ a panda table
    """
    # print(dft.duplicated())  # find repeat lines
    dft = read_file(filename)
    print(dft.drop_duplicates())  # delete repeat lines
    dft_ = dft.drop_duplicates([i for i in range(line)])
    print(dft_)
    return dft_


def save_file(save_path):
    """
    保存数据集到txt
    :param save_path: Where U put data in the dir
    :return: None
    """
    data = delete_repeat()
    np.savetxt(save_path, data)
    # dft_.to_csv('dataset_v00.txt', sep=' ')


def array2pic(path):
    """
    数组转图像格式
    :param path: Where U put data in the dir
    :return: a image
    """
    img = mpimg.imread(path)  # 这里读入的数据是 float32 型的，范围是0-1
    # 方法转换之后显示效果不好
    img = Image.fromarray(np.uint8(img))
    img.show()


def slide_array(path, output_path):
    """
    减小数据集
    :param: path: Where U put data in the dir
    :param: output_path: Where U save data in the dir
    :return: None
    """
    tmp = np.loadtxt(path)
    tmp = tmp[0:100000, :]
    np.savetxt(output_path, tmp)


def add_row(path, output_path):
    """
    对array增加行或列
    :param: path: Where U put data in the dir
    :param: output_path: Where U save data in the dir
    :return: None
    """
    tmp = np.loadtxt(path)
    # tmp_data = tmp[:, 0:-1]
    # tmp_label = tmp[:, -1]
    col = np.array([0] * len(tmp))
    print(np.shape(col))
    for i in range(32):
        tmp = np.column_stack((tmp, col))
    # tmp_data = np.column_stack((tmp_data, tmp_label))
    np.savetxt(output_path, tmp,fmt='%d')


def array_detail(path, output_path):
    """
    计算array的行列,统计文件
    :param: path: Where U put data in the dir
    :param: output_path: Where U save data in the dir
    :return: None
    """
    cwd = os.getcwd()
    NUM_FLIES = 0  # 计数器
    FILENAMES = []  # 文件名
    NUM_SAMPLES = []  # 样本数
    NUM_FEATURES = []  # 特征数
    classes = os.listdir(cwd + '/' + path)  # label的文件列表
    for name in classes:
        class_path = cwd + '/' + path + '/' + name + "/"
        # print(class_path)
        if os.path.isdir(class_path):
            for datafile in os.listdir(class_path):
                if os.path.splitext(datafile)[1] == '.txt':
                    NUM_FLIES = NUM_FLIES + 1
                    print(class_path + datafile)
                    tmp = np.loadtxt(class_path + datafile)
                    FILENAMES.append(datafile)
                    NUM_SAMPLES.append(len(tmp))
                    NUM_FEATURES.append(len(tmp[0]))
                    # tmp_t = np.array(FILENAMES+NUM_SAMPLES+NUM_FEATURES)
    tmp_0 = np.array(FILENAMES)
    tmp_1 = np.array(NUM_SAMPLES)
    tmp_2 = np.array(NUM_FEATURES)
    tmp_0 = tmp_0.reshape(-1, 1)
    tmp_1 = tmp_1.reshape(-1, 1)
    tmp_2 = tmp_2.reshape(-1, 1)
    tmp_t = np.concatenate((tmp_0, tmp_1, tmp_2), axis=1)
    np.savetxt(output_path, tmp_t)


def concat_data(input_dir, output):
    """
    对于数组形式的数据进行拼接
    :param input1: 操作文件的位置
    :param output: 输出地址
    :return:
    """
    cwd = os.getcwd()  # 获取代码所在位置
    data_path = cwd + '/' + input_dir  # 获取数据集的绝对路径
    tmp = np.loadtxt(output)  # 数组变量
    for datafile in os.listdir(data_path):
        # 对于文件夹下的每一个数据集，操作
        datafile_t = data_path + '/' + datafile
        tmp_in = np.loadtxt(datafile_t)
        print(datafile_t)
        if os.path.splitext(datafile)[1] == '.txt':
            # 判断是否为txt数据集
            if len(tmp) == 0:
                # 判定输出的数据集是否为空
                tmp = np.loadtxt(datafile_t)
                seed = int(len(tmp)/2)
                tmp = tmp[seed:, :]
            elif len(tmp_in[0]) == len(tmp[0]):
                # 保证features相等
                seed = int(len(tmp_in) / 2)
                tmp_in = tmp_in[seed:, :]
                tmp = np.concatenate((tmp, tmp_in), axis=0)
            else:
                print('Features are NOT SAME!')
    print('OUTPUT:' + output)
    np.savetxt(output, tmp, fmt='%d')


def shuffle(path):
    """
    打乱array
    :param path: Where U put data in the dir
    :return:
    """
    X = np.loadtxt(path)
    y = X[:, -1].astype(np.int)
    X = X[:, :-1]
    rs = ShuffleSplit(n_splits=1, test_size=.25, random_state=0)
    rs.get_n_splits(X)
    # print(rs)
    for train_index, test_index in rs.split(X, y):
        # print("Train Index:", train_index, ",Test Index:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # print(X_train,X_test,y_train,y_test)
    print("==============================")
    print("Making dataset")
    # rs = ShuffleSplit(n_splits=3, train_size=.5, test_size=.25, random_state=0)
    np.savetxt('X_train_' + path, X_train, fmt='%d')
    np.savetxt('Y_train_' + path, y_train, fmt='%d')
    np.savetxt('X_test_' + path, X_test, fmt='%d')
    np.savetxt('Y_test_' + path, y_test, fmt='%d')
    # return X_train, X_test, y_train, y_test
    print("==============================")
    print('FINISHED !')



def process_line(line):
    """
    根据文件名生成一个队列 
    :param: filename Where U put data in the dir
    :return: img, label
    """
    tmp = [int(val) for val in line.strip().split(',')]
    x = np.array(tmp[:-1])
    y = np.array(tmp[-1:])
    return x, y


def generate_arrays_from_file(path, batch_size):
    while 1:
        f = open(path)
        cnt = 0
        X = []
        Y = []
    for line in f:
        # create Numpy arrays of input data
        # and labels, from each line in the file
        x, y = process_line(line)
        X.append(x)
        Y.append(y)
        cnt += 1
        if cnt == batch_size:
            cnt = 0
            yield (np.array(X), np.array(Y))
            X = []
            Y = []
    f.close()


# filename = 'small_train.txt'
# out_file = 'small_train_18.txt'
# # slide_array(filename,out_file)
# read_file(out_file)
# filename = 'mj_data/ZHIYI_v2/non_king'
# filename = '/home/tribody/My_RMTTF/My_work_Z0/mj_data/ZHIYI_v2/non_king/'
# outfile = 'tt0.txt'
outfile = 'mj_data/non_king_processed/X_train_non_king_2.txt'
outfile_0 = 'mj_data/non_king_processed/X_train_non_king_2_pic.txt'
add_row(outfile,outfile_0)
# concat_data(filename,outfile)
# tt = np.loadtxt(outfile).astype(np.float32)
# shuffle(outfile)
# read_file(outfile)
# slide_array(outfile, '1_' + outfile)
