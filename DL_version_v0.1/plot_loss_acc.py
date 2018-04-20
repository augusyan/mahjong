# -*- coding:utf-8 -*-
"""
@author : Augus Yan
@file : plot_loss_acc.py
@time : 2018/4/18 15:59
@function : 
"""
import numpy as np
import matplotlib.pyplot as plt
import ast


def load_log(filename):
    # input = open('res50.log', 'r')
    with open(filename, 'r') as load_f:
        model_name = load_f.read()
    # print(input)
        model_name = ast.literal_eval(model_name)
    val_loss = model_name['val_loss']
    loss = model_name['loss']
    val_acc = model_name['val_acc']
    acc = model_name['acc']
    return val_loss, loss, val_acc, acc

# load data
res50_val_loss, res50_loss, res50_val_acc, res50_acc = load_log('res50.log')
fc_val_loss, fc_loss, fc_val_acc, fc_acc = load_log('sysv2_u1200e100d3l2r.log')
res1001_val_loss, res1001_loss, res1001_val_acc, res1001_acc = load_log('res1001e0-40.log')
liner_val_loss, liner_loss, liner_val_acc, liner_acc = load_log('liner_e50adam.log')

# start figure
plt.figure('Training curves on Mahjong Dataset')
plt.subplot(121)
# res50 model
plt.plot(res50_val_loss, color='blue', linewidth=1.0, linestyle='--', label='Res50_val_loss')
plt.plot(res50_loss, color='blue', linewidth=1.0, linestyle='-', label='Res50_train_loss')
# dnn3 model
plt.plot(fc_val_loss[:50], color='coral', linewidth=1.0, linestyle='--', label='FC_val_loss')
plt.plot(fc_loss[:50], color='coral', linewidth=1.0, linestyle='-', label='FC_train_loss')
# res1001 model
plt.plot(res1001_val_loss, color='brown', linewidth=1.0, linestyle='--', label='Res1001_val_loss')
plt.plot(res1001_loss, color='brown', linewidth=1.0, linestyle='-', label='Res1001_loss')
# liner model
plt.plot(liner_val_loss, color='g', linewidth=1.0, linestyle='--', label='liner_val_loss')
plt.plot(liner_loss, color='g', linewidth=1.0, linestyle='-', label='liner_loss')
# 设置坐标轴刻度
my_x_ticks = np.arange(0, 50, 5)
my_y_ticks = np.arange(0.1, 3, 0.2)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.legend(loc='best')

plt.subplot(122)
# res50 model
plt.plot(res50_val_acc, color='blue', linewidth=1.0, linestyle='--', label='Res50_val_acc')
plt.plot(res50_acc, color='blue', linewidth=1.0, linestyle='-', label='Res50_train_acc')
# dnn3 model
plt.plot(fc_val_acc[:50], color='coral', linewidth=1.0, linestyle='--', label='FC_val_acc')
plt.plot(fc_acc[:50], color='coral', linewidth=1.0, linestyle='-', label='FC_train_acc')
# GoResnet
plt.plot(res1001_val_acc, color='brown', linewidth=1.0, linestyle='--', label='Res1001_val_acc')
plt.plot(res1001_acc, color='brown', linewidth=1.0, linestyle='-', label='Res1001_acc')
# liner model
plt.plot(liner_val_acc, color='g', linewidth=1.0, linestyle='--', label='liner_val_acc')
plt.plot(liner_acc, color='g', linewidth=1.0, linestyle='-', label='liner_acc')
# 设置坐标轴刻度
my_x_ticks = np.arange(0, 50, 5)
my_y_ticks = np.arange(0.2, 1, 0.05)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.legend(loc='best')
# plt.grid(True)

# plt.subplot(212)
# plt.grid(True)

plt.show()
