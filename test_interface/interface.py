# -*- coding: utf-8 -*-
"""
Created on Sun Nov 05 17:26:14 2017

@author: ren
"""

from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import os
import datetime
from sklearn import cross_validation
import os, json, copy
import sys
import write_feature

dict = {}

inits = input("请输入初始手牌,每张牌用，隔开:\n")
str_array = inits.split(',')
init_cards = []
for i in str_array:
    init_cards.append(int(i))
# 输入正确性判断
while len(init_cards) != 13 and len(init_cards) != 14:
    print('输入有误，请重新输入\n')
    inits = input("请输入初始手牌:\n")
    str_array = inits.split(',')
    init_cards = []
    for i in str_array:
        init_cards.append(int(i))

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
        init_cards[int(index)] = int(card)

dict["init_cards"] = init_cards
actions = []

'''
{"combine_cards": [],
 "seat_id": 2, 
 "op_card": 255,
  "action_type": "A",
   "out_seat_id": 255}
    
'''

king_card = input('请输入本局的宝牌\n')
flag2 = True
while flag2:
    f = input('输入确认y/n\n')
    if f == 'y':
        flag2 = False
    elif f == 'n':
        king_card = input('请输入本局的宝牌\n')
king_cards = [int(king_card)]
dict['king_cards'] = king_cards

print("请输入抓牌出牌的牌，抓牌用g开头，出牌用d开头，例如抓1万，g01,出1万，d01")
while 1:
    inputs = input("牌\n")
    action = {
        "combine_cards": [],
        "seat_id": 0,
        "action_type": "g",
        "out_seat_id": 255
    }
    if inputs[0] == "G" or inputs[0] == "g":
        action["combine_cards"] = []
        action["seat_id"] = 0
        action["op_card"] = int(inputs[1]) * 16 + int(inputs[2])
        action["action_type"] = "G"
        action["out_seat_id"] = 255
        actions.append(action)
        dict["actions"] = actions
        with open("./input", 'w') as json_file:
            json.dump(dict, json_file)
        fo = open("./output.txt", "w")  # 写的文件
        fo.truncate()  # 清空output
        with open("./input", 'r') as json_file:  # 打开文件
            data = json.load(json_file)
            write_feature.write(fo, data)
        fo.close()

    elif inputs[0] == "d" or inputs[0] == "D":
        action["combine_cards"] = []
        action["seat_id"] = 0
        action["op_card"] = int(inputs[1]) * 16 + int(inputs[2])
        action["action_type"] = "d"
        action["out_seat_id"] = 255
        actions.append(action)
        dict["actions"] = actions
        with open("./input", 'w') as json_file:
            json.dump(dict, json_file)

    elif inputs[0] == "A" or inputs[0] == "a":
        action["combine_cards"] = []
        action["seat_id"] = 0
        action["op_card"] = 255
        action["action_type"] = "A"
        action["out_seat_id"] = 255
        actions.append(action)
        dict["actions"] = actions
        with open("./input", 'w') as json_file:
            json.dump(dict, json_file)
        break
    else:
        print("请重新输入\n")
        continue
