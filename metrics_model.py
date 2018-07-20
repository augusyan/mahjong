# -*- coding: utf-8 -*-
# @Time    : 18-6-21 上午10:26
# @Author  : Yan
# @Site    : 
# @File    : metrics_model.py
# @Software: PyCharm Community Edition
# @Function: 
# @update:


import keras
import time
import os
import sys
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.callbacks import TensorBoard
from keras.applications.imagenet_utils import decode_predictions
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support, \
    precision_recall_curve, roc_auc_score, classification_report
from sklearn import metrics


class Metrics(object):
    """
    Main function to metric the preference of muti or binary classification model
    """
    def __init__(self, model_path, data_path, label_path, classes):
        self.model_path = model_path    # h5model path
        self.data_path = data_path      # test data path
        self.label_path = label_path    # label path or just split from test data
        self.classes = classes          # muti classes

        self.y_pred, self.y_pred_prob,self.y_true, self.y_true_oh = self.load_data()

    def metrics_mutil(self):
        """
        Mainly figure of muti-class model,including precision recall f1 cm etc.
        :return:
        """
        pass

    def load_data(self):
        """
        load test data from path
        :return:
        """
        # time record
        t0 = time.clock()
        model = load_model(self.model_path)
        print('Model Restore! ', time.clock() - t0)
        t0 = time.clock()
        x_test = np.loadtxt(self.data_path, delimiter=' ',
                            dtype=np.int32)

        # To reshape the input to a square size
        if np.mod(len(x_test[0]), np.sqrt(len(x_test[0]))) == 0.0:
            x_test_pic = int(np.sqrt(len(x_test[0])))
        else:
            x_test_pic = int(np.sqrt(len(x_test[0]))) + 1
        # x_test_pic = 16
        x_test = x_test.reshape((-1, 1, x_test_pic, x_test_pic))
        x_test = x_test[:, :]

        y_test = np.loadtxt(self.label_path, delimiter=' ', dtype=np.int32)
        y_test = y_test[:]
        print('Data loaded! ', time.clock() - t0)
        # load model to give a predict probability distribution
        y_pred_prob = model.predict(x_test, batch_size=1, verbose=0)

        # cal classes of probability distribution
        y_pred = []
        for i in range(len(y_pred_prob)):
            y_pred.append(np.argmax(y_pred_prob[i]))
        y_pred = np.array(y_pred)
        y_true = y_test
        print("y_pred and y_true generated!")

        # give a one-hot code to test data
        y_test_oh = np_utils.to_categorical(y_test, num_classes=self.classes)
        return y_pred, y_pred_prob, y_true, y_test_oh

    def cls_report(self):
        """
        give a table of p,r,f1 etc.
        :return:
        """
        target_names = [str(i) for i in range(self.classes)]
        print(classification_report(self.y_true, self.y_pred, target_names=target_names))

    def plotPR_m(self):
        """
        Using one-hot code to draw the muti-classes precision_recall_curve
        :return:
        """
        # print('调用函数auc：', metrics.roc_auc_score(self.y_test_oh, self.y_pred_prob, average='micro'))
        # 首先将矩阵y_one_hot和y_score展开，然后计算假正例率FPR和真正例率TPR
        p, r, thresholds = metrics.precision_recall_curve(self.y_true_oh.ravel(), self.y_pred_prob.ravel())
        # auc = metrics.auc(r, p)
        # print('手动计算auc：', auc)

        # 绘图
        # plt.rcParams['font.sans-serif'] = u'SimHei'
        # plt.rcParams['axes.unicode_minus'] = False
        # FPR就是横坐标,TPR就是纵坐标
        plt.plot(r, p, c='r', lw=2, alpha=0.7)
        plt.plot((1, 0), (1, 0), c='#808080', lw=1, ls='--', alpha=0.7)
        plt.xlim((-0.01, 1.02))
        plt.ylim((-0.01, 1.02))
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlabel('Recalls Rate', fontsize=13)
        plt.ylabel('Precisions Rate', fontsize=13)
        plt.grid(b=True, ls=':')
        plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
        plt.title('PR Curve', fontsize=17)
        plt.show()

    def plotAUC_m(self):
        """
        Using one-hot code to draw the muti-classes AUC_curve
        :return:
        """
        print('调用函数auc：', metrics.roc_auc_score(self.y_true_oh, self.y_pred_prob, average='micro'))

        # 2、手动计算micro类型的AUC
        # 首先将矩阵y_one_hot和y_score展开，然后计算假正例率FPR和真正例率TPR
        fpr, tpr, thresholds = metrics.roc_curve(self.y_true_oh.ravel(), self.y_pred_prob.ravel())
        auc = metrics.auc(fpr, tpr)
        print('手动计算auc：', auc)

        # 绘图
        plt.rcParams['font.sans-serif'] = u'SimHei'
        plt.rcParams['axes.unicode_minus'] = False
        # FPR就是横坐标,TPR就是纵坐标
        plt.plot(fpr, tpr, c='r', lw=2, alpha=0.7, label=u'AUC=%.3f' % auc)
        plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
        plt.xlim((-0.01, 1.02))
        plt.ylim((-0.01, 1.02))
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlabel('False Positive Rate', fontsize=13)
        plt.ylabel('True Positive Rate', fontsize=13)
        plt.grid(b=True, ls=':')
        plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
        plt.title('ROC and AUC', fontsize=17)
        plt.show()


def model_test(model_path, target):
    """
    Test model
    :param model_path:
    :param X:
    :return:
    """
    # 忽略硬件加速的警告信息
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # file_path = 'images/0a70f64352edfef4c82c22015f0e3a20.jpg'
    #
    # img = image.load_img(file_path, target_size=(224, 224))
    # x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    #
    t0 = time.clock()
    model = load_model(model_path)

    y = model.predict(target)
    # print(np.argmax(y))

    # print('Predicted:', decode_predictions(y, top=3)[0])
    model = load_model(model_path)  # 载入模型
    t0 = t0 - time.clock()
    print('Model Restore! Using time:', t0)
    print(model.predict_classes(y, batch_size=1, verbose=0))


def plotPR(recalls, precision):
    """

    :param recalls:
    :param precision:
    :return:
    """
    plt.figure()
    # PR
    plt.plot(recalls, precision, 'r-', label='train acc')
    plt.grid(True)
    plt.xlabel('Recalls', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.show()


# X = model.predict(x_test, batch_size=1, verbose=0)
# print(X)
# print("X size|", np.shape(X))
#
# pred_X_classes = []
# pred_X_p = []
# for i in range(len(X)):
#     pred_X_classes.append(np.argmax(X[i]))
# predict_X = np.array(pred_X_classes)
#
# print("predict_X",predict_X)
# Y = y_test[:2000]
# print("Y", Y)
#
# print("predict_X size|", np.shape(predict_X))
#
# # X = np_utils.to_categorical(X, num_classes=8)
# # predict_X = np_utils.to_categorical(predict_X, num_classes=8)
# # 分别计算precision, recall, f1score,
# precision, recall, f1score, support = precision_recall_fscore_support(
#     Y, predict_X, [0,1,2,3,4,5,6,7],average="micro")
# # 利用多分类转二分类画PR图
# # p,r,t = precision_recall_curve(Y, X[:,0])
# print("precision|",precision)
# print("recall|",recall)
# print("F1 score|",f1score)
# # print("support|",support)
#
# plotPR(recall,precision)
# # print(np.shape(X))
# # print(predict_X+1)
# # print(X[0][predict_X+1])


if __name__ == '__main__':
    model_path = 'model/20180528/sys_non_king_op_resnet_simple_e50d5.h5'
    data_path = 'mj_data/data_201804/non_king_op_20180528/output_X_test_pic.txt'
    label_path = 'mj_data/data_201804/non_king_op_20180528/output_Y_test.txt'
    classes = 8
    m = Metrics(model_path, data_path, label_path, classes)
    m.plotAUC_m()
    m.cls_report()
    m.plotPR_m()


