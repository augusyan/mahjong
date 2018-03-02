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
    tmp = tmp[0:90000,:]
    np.savetxt(output_path, tmp)


def add_row(path, output_path):
    """
    对array增加行或列
    :param: path: Where U put data in the dir
    :param: output_path: Where U save data in the dir
    :return: None
    """

    tmp = np.loadtxt(path)
    tmp_data = tmp[:,0:-1]
    tmp_label = tmp[:,-1]
    col = np.array([0]*len(tmp))
    print(np.shape(col))
    for i in range(28):
        tmp_data = np.column_stack((tmp_data,col))
    tmp_data = np.column_stack((tmp_data, tmp_label))
    np.savetxt(output_path, tmp_data)


def mix_array(path, output_path):
    """
    对array增加行或列
    :param: path: Where U put data in the dir
    :param: output_path: Where U save data in the dir
    :return: None
    """
    

def shuffle(path):
    """
    打乱array
    :param path: Where U put data in the dir
    :return:
    """
    pass


def process_line(line):
    """
    根据文件名生成一个队列 
    :param filename: Where U put data in the dir
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


filename = 'small_train.txt'
out_file = 'small_train_18.txt'
# slide_array(filename,out_file)
add_row(filename,out_file)
read_file(out_file)

