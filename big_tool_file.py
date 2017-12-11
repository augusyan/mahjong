# -*-coding:utf-8 -*-
# __author__='Yan'
# function: rename file, chunk read/write
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn import svm

# global setting
pd.set_option('display.max_columns', None)


# 零均值化
def zeroMean(dataMat):
    meanVal = np.mean(dataMat, axis=0)  # 按列求均值，即求各个特征的均值
    newData = dataMat - meanVal
    return newData, meanVal


# pca降维预处理
def pca(dataMat, n):
    newData, meanVal = zeroMean(dataMat)
    covMat = np.cov(newData, rowvar=0)  # 求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本

    eigVals, eigVects = np.linalg.eig(np.mat(covMat))  # 求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
    eigValIndice = np.argsort(eigVals)  # 对特征值从小到大排序
    n_eigValIndice = eigValIndice[-1:-(n + 1):-1]  # 最大的n个特征值的下标
    n_eigVect = eigVects[:, n_eigValIndice]  # 最大的n个特征值对应的特征向量
    lowDDataMat = newData * n_eigVect  # 低维特征空间的数据
    reconMat = (lowDDataMat * n_eigVect.T) + meanVal  # 重构数据
    return lowDDataMat, reconMat


# 重命名文件夹里的文件
def rename_file():
    path = "notMNIST_small/A/"
    count = 0
    for file in os.listdir(filename):
        # if os.path.isfile(os.path.join(path,file))==True:
        # if file.find('.png') == 0:
        newname = str(count) + '.png'
        os.rename(os.path.join(filename, file), os.path.join(path, newname))
        count += 1
        print(file)
        # print(file)


# 从txt/csv格式读取文件
def read_file(path):
    # read csv
    # dfc = pd.read_csv(path)
    # read txt
    # dft = np.loadtxt(path)

    dft = pd.read_table(path, header=None, delim_whitespace=True)
    # # data = pd.DataFrame(dft,columns=['col1','col2','col3','col4','col5','col6','col7','col8','col9',
    #                                  #'col10','col11','col12','col13','col14','col15','col16','col17','col18','col19',
    #                                  #'col20','col21', 'col22', 'col23', 'col24', 'col25', 'col26', 'col27', 'col28',
    #                                  #'col29','col30','label'])
    # loop = True
    # chunkSize = 100000
    # chunks = []
    # while loop:
    #     try:
    #         chunk = dft.get_chunk(chunkSize)
    #         distrbution = chunk.describe()
    #         print(distrbution)
    #         chunks.append(chunk)
    #     except StopIteration:
    #         loop = False
    #         print("Iteration is stopped.")
    #df = pd.concat(chunks, ignore_index=True)
    distrbution = dft.describe()
    print(distrbution)
    return dft


# 删除重复的行
def delete_repeat(line=23):
    # print(dft.duplicated())            # find repeat lines
    #dft_ = dft.drop([6,7,8,23,24,25,26], axis=1, inplace=False)       # delete col
    #distrbution = dft_.describe()
    #print(dft.values[:, 29])
    #print(dft.count)
    dft = read_file(filename)
    print(dft.drop_duplicates())            # delete repeat lines
    dft_ = dft.drop_duplicates([i for i in range(line)])
    print(dft_)
    return dft_


# 删除重复的行,保存到txt文件
def save_file():
    data = delete_repeat()
    np.savetxt('dataset_v11.txt', data)
    # dft_.to_csv('dataset_v00.txt', sep=' ')


# txt to csv
def txt2csv(filename):
    # txt 2 csv
    txt = np.loadtxt(filename)
    txtDF = pd.DataFrame(txt)
    txtDF.to_csv(filename, index=False)


# 数据集简单min/max处理
def datatxt_nor_test(trainData):
    tmp = trainData
    amin, amax = trainData.min(), trainData.max()
    print('---')
    # print (amin, amax)
    trainData = (trainData - amin) / (amax - amin)

    print(trainData)


# 数据集归一化处理
def datatxt_normalization(trainData, type='mean'):
    tmp = trainData
    if type == 'mean':
        min_max_scaler = preprocessing.MinMaxScaler()
        trainData = min_max_scaler.fit_transform(trainData)

    if type == 'standard':
        standard_scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
        trainData = standard_scaler.fit_transform(trainData)

    if type == 'normal':
        normalized = preprocessing.normalize(norm='l2')
        trainData = normalized.fit_transform(trainData)

    trainData[:, -1] = tmp[:, -1]
    print('---')
    print(trainData)
    return trainData


# numpy方式读取txt,归一化
def read_datatxt_processing(filename):
    train_Data = np.genfromtxt(filename, delimiter=' ')
    # print(trainData)
    # datatxt_nor_test(trainData)
    datatxt_normalization(train_Data)


# k-fold生成训练集和测试集
# def k_fold_set(input, test_size=0.2):
#     print(input)
#     input_x = input[:,0:-1]
#     input_y = input[:,-1]
#     input = KFold(n_splits=10, shuffle=True, random_state=None)
#     # train_data = np.column_stack(X_train, y_train)
#     # test_data = np.column_stack(X_test, y_test)
#     np.savetxt(train_path, train_data)
#     np.savetxt(test_path, test_data)


# csv格式转换 tfrecord,需要大内存和线程控制
def csv2tfrecord(filename,output_file):
    # load .csv file
    # train_frame = pd.read_csv(filename, header=None, delim_whitespace=True)
    train_frame = np.loadtxt(filename)
    # print(train_frame.head())
    # train_labels_frame = train_frame[-1]
    # print(train_labels_frame.shape)

    train_values = train_frame[:,0:-1]
    train_labels_values = train_frame[:,-1].astype(np.int32)
    train_size = train_values.shape[0]
    # ------------------create TFRecord file------------------------ #
    writer = tf.python_io.TFRecordWriter(output_file)
    for i in range(train_size):
        image_raw = train_values[i].tostring()
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "image_raw": tf.train.Feature(bytes_list=tf.train.FloatList(value=[image_raw])),
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[train_labels_values[i]]))
                }
            )
        )
        writer.write(record=example.SerializeToString())
    writer.close()


def make_train_test(filename,train_path,test_path):
    data = np.loadtxt(filename).astype(int)
    # length = int(data.shape[0] * 0.8)
    tmp = data[0:2240000]
    np.savetxt(train_path, tmp)
    tmp = data[2240000:-1]
    np.savetxt(test_path,tmp)
    print('Making trian and test file done!')


# 主函数
# ------------------PATH------------------ #
# train_record = 'mj_dataset/dataset_v11_mini.tfrecords'
# train_record = 'mj_dataset/data223_train.tfrecords'
filename = 'mj_dataset/non_kings3.txt'
# filename = 'mj_dataset/output1.txt'
train_path = 'mj_dataset/non_kings'+'_train_3.txt'
test_path = 'mj_dataset/non_kings'+'_test_3.txt'
# processed_path = 'mj_dataset/data223'+'_p.txt'
# ------------------FUNCITONS------------------ #
#trainData = np.genfromtxt(filename, delimiter=' ')
#a = datatxt_normalization(trainData, 'mean')
#print('| Processing Data |')
#print(a)

# save processed file
#np.savetxt(processed_path, a)

#print('| Analysis Data |')
# csv2tfrecord(filename,train_record)
# print(read_file(filename))
make_train_test(filename, train_path, test_path)








