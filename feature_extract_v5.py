# -*- coding:utf-8 -*-

import copy
import sys
# 1通常手向听数
def wait_types_comm(tile_list):
    common_waiting = {'common_waiting0': 0,
                      'common_waiting1': 0,
                      'common_waiting2': 0,
                      'common_waiting3': 0,
                      'common_waiting4': 0,
                      'common_waiting5': 0,
                      'common_waiting6': 0,
                      'common_waiting7': 0,
                      'common_waiting8': 0,
                      'common_waiting9': 0,
                      'common_waiting10': 0,
                      'common_waiting11': 0,
                      'common_waiting12': 0,
                      'common_waiting13': 0,
                      }
    tempList = tile_list
    # print tempList
    wait_num = 14
    '''
    # 考虑副露的情况
    suits_len = len(suits)
    wait_num -= suits_len * 3
    '''
    sz = 0  # 顺子数
    kz = 0  # 刻子数
    dzk = 0  # 搭子 aa
    dzs12 = 0  # 搭子ab
    dzs13 = 0  # 搭子ac
    # 判断顺子数
    i = 0
    while i <= len(tempList) - 3:
        if tempList[i] & 0xF0 != 0x30 and tempList[i] + 1 in tempList and tempList[i] + 2 in tempList:
            # print(tempList[i], "i")
            wait_num -= 3
            sz += 1
            card0 = tempList[i]
            card1 = tempList[i] + 1
            card2 = tempList[i] + 2
            tempList.remove(card0)
            tempList.remove(card1)
            tempList.remove(card2)
        else:
            i += 1
    # print(tempList)

    # 判断刻子数
    j = 0
    while j <= len(tempList) - 3:
        if tempList[j + 1] == tempList[j] and tempList[j + 2] == tempList[j]:
            # print(tempList[j], "j")
            wait_num -= 3
            kz += 1
            card = tempList[j]
            tempList.remove(card)
            tempList.remove(card)
            tempList.remove(card)
        else:
            j += 1
    # print(tempList)

    # 判断搭子aa
    x = 0
    while x <= len(tempList) - 2:
        # print(tempList[x], "x")
        if tempList[x + 1] == tempList[x]:
            dzk += 1
            # wait_num -=2
            card = tempList[x]
            tempList.remove(card)
            tempList.remove(card)
        else:
            x += 1

    # 判断搭子ab ac
    k = 0
    while k <= len(tempList) - 2:
        if tempList[k] & 0xF0 != 0x30:
            # print(tempList[k], "k")
            if tempList[k] + 1 in tempList:
                # wait_num -= 2
                dzs12 += 1
                card0 = tempList[k]
                card1 = tempList[k] + 1
                tempList.remove(card0)
                tempList.remove(card1)
            elif tempList[k] + 2 in tempList:
                # wait_num -= 2
                dzs13 += 1
                card0 = tempList[k]
                card2 = tempList[k] + 2
                tempList.remove(card0)
                tempList.remove(card2)
            else:
                k += 1
        else:
            k += 1
    if dzk > 0:  # 如果搭子aa>0 ,取其中一个作为将牌，并且向听数-2
        wait_num -= 2
        if dzk - 1 + dzs12 + dzs13 - (4 - sz - kz) <= 0:  # 如果搭子加面子<=4,向听数再减去搭子数*2
            wait_num -= (dzk - 1 + dzs12 + dzs13) * 2
        else:
            wait_num -= (4 - sz - kz) * 2  # 否则 向听数只减去多余的，即向听数减到为0
    else:  # 如果搭子aa=0，取一张单牌作为将的候选，向听数-1
        wait_num -= 1
        if dzk + dzs12 + dzs13 - (4 - sz - kz) <= 0:  # 向上同理
            wait_num -= (dzk + dzs12 + dzs13) * 2
        else:
            wait_num -= (4 - sz - kz) * 2
    # print(tempList)

    common_waiting['common_waiting' + str(wait_num)] = 1
    return wait_num
    # print(common_waiting)


# 2 七对的向听数判断
def wait_types_7(tile_list):
    wait_7couples = {
        'seven_waiting0': 0,
        'seven_waiting1': 0,
        'seven_waiting2': 0,
        'seven_waiting3': 0,
        'seven_waiting4': 0,
        'seven_waiting5': 0,
        'seven_waiting6': 0,
        'seven_waiting7': 0,
    }
    wait_num = 7  # 表示向听数
    tile_list.sort()  # L是临时变量，传递tile_list的值
    L = set(tile_list)
    for i in L:
        # print("the %d has %d in list" % (i, tile_list.count(i)))
        if tile_list.count(i) >= 2:
            wait_num -= 1
    # print(tile_list)
    # wait_types_7['seven_waiting'+str(wait_num)] = 1
    # print(wait_num)
    wait_7couples['seven_waiting' + str(wait_num)] = 1
    # print(wait_7couples)
    return wait_num


# 3 十三浪的向听数判断
def wait_types_13(tile_list):
    # 十三浪的向听数判断，手中十四张牌中，序数牌间隔大于等于3，字牌没有重复所组成的牌形
    # 先计算0x0,0x1,0x2中的牌，起始位a，则a+3最多有几个，在wait上减，0x3计算不重复最多的数
    wait_13lan = {
        'thirteen_waiting0': 0,
        'thirteen_waiting1': 0,
        'thirteen_waiting2': 0,
        'thirteen_waiting3': 0,
        'thirteen_waiting4': 0,
        'thirteen_waiting5': 0,
        'thirteen_waiting6': 0,
        'thirteen_waiting7': 0,
        'thirteen_waiting8': 0,
        'thirteen_waiting9': 0,
        'thirteen_waiting10': 0,
        'thirteen_waiting11': 0,
        'thirteen_waiting12': 0,
        'thirteen_waiting13': 0,
        'thirteen_waiting14': 0,
    }
    wait_num = 14  # 表示向听数
    max_num_wait = 0
    # print(wait_13lan)
    L = set(tile_list)  # 去除重复手牌
    L_num0 = []  # 万数牌
    L_num1 = []  # 条数牌
    L_num2 = []  # 筒数牌
    for i in L:
        if i & 0xf0 == 0x30:
            # 计算字牌的向听数
            wait_num -= 1
        if i & 0xf0 == 0x00:
            L_num0.append(i & 0x0f)
        if i & 0xf0 == 0x10:
            L_num1.append(i & 0x0f)
        if i & 0xf0 == 0x20:
            L_num2.append(i & 0x0f)
    wait_num -= calculate_13(L_num0)
    # 减去万数牌的向听数
    wait_num -= calculate_13(L_num1)
    # 减去条数牌的向听数
    wait_num -= calculate_13(L_num2)
    # 减去筒数牌的向听数
    # print(L)
    # print(L_num0)
    # print(L_num1)
    # print(L_num2)
    # print(wait_num)
    wait_13lan['thirteen_waiting' + str(wait_num)] = 1
    # print(wait_13lan)
    return wait_num


# 4 九幺的向听数判断
def wait_types_19(tile_list):
    # 九幺的向听数判断，由一、九这些边牌、东、西、南、北、中、发、白这些风字牌中的任意牌组成的牌形。以上这些牌可以重复
    wait_19 = {
        'one_nine_waiting0': 0,
        'one_nine_waiting1': 0,
        'one_nine_waiting2': 0,
        'one_nine_waiting3': 0,
        'one_nine_waiting4': 0,
        'one_nine_waiting5': 0,
        'one_nine_waiting6': 0,
        'one_nine_waiting7': 0,
        'one_nine_waiting8': 0,
        'one_nine_waiting9': 0,
        'one_nine_waiting10': 0,
        'one_nine_waiting11': 0,
        'one_nine_waiting12': 0,
        'one_nine_waiting13': 0,
        'one_nine_waiting14': 0,
    }
    wait_num = 14  # 表示向听数
    # tile_list.sort()  # 排序
    # L = set(tile_list)  # L是临时变量，传递tile_list的值
    for i in tile_list:
        if i & 0x0f == 0x01 or i & 0x0f == 0x09 or i & 0xf0 == 0x30:
            wait_num -= 1
    wait_19['one_nine_waiting' + str(wait_num)] = 1
    # print(wait_19)
    return wait_num


def calculate_13(tiles):
    # 计算十三浪的数牌最大向听数
    if len(tiles) == 0:
        return 0
    if len(tiles) == 1:
        return 1
    if len(tiles) == 2:
        if tiles[0] + 3 <= tiles[1]:
            return 2
        else:
            return 1
    if len(tiles) >= 3:
        return max((tiles.count(1) + tiles.count(4) + tiles.count(7)),
                   (tiles.count(1) + tiles.count(4) + tiles.count(8)),
                   (tiles.count(1) + tiles.count(4) + tiles.count(9)),
                   (tiles.count(1) + tiles.count(5) + tiles.count(8)),
                   (tiles.count(1) + tiles.count(5) + tiles.count(9)),
                   (tiles.count(1) + tiles.count(6) + tiles.count(9)),
                   (tiles.count(2) + tiles.count(5) + tiles.count(8)),
                   (tiles.count(2) + tiles.count(5) + tiles.count(9)),
                   (tiles.count(2) + tiles.count(6) + tiles.count(9)),
                   (tiles.count(3) + tiles.count(6) + tiles.count(9)))

# 5-9 万，条，筒，风，箭个数

def numOfCards(handcards):
    feature = [0,0,0,0,0]
    for d in handcards:
        if (d & 0xF0) / 16 < 3:
            feature[int((d & 0xF0) / 16)] += 1
        elif (d & 0xF0) / 16 == 3 and (d & 0x0F) < 5:
            feature[3] += 1
        else:
            feature[4] += 1
    #print feature
    return feature

# # 10 -43 万的AA



def numofcard(handcards):
  feature = [0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
]
  for d in handcards:
      num=d & 0x0F
      # 1-9万 【0-8】
      if (d & 0xF0)/16 == 0:
          feature[num-1]+=1
      # 1-9条  【9-17】
      elif(d & 0xF0)/16 == 1:
          feature[8+ num] += 1
      # 1-9饼 【18-26】
      elif(d & 0xF0)/16 == 2:
          feature[17 + num] += 1
      # 字牌 【27-33】
      elif (d & 0xF0) / 16 == 3:
          #print handcards
          #print num
          feature[26 + num] += 1
  #print feature
  return feature
#44-67
def ab(handcards):
    feature=[
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    k =0
    while k <= len(handcards) - 2:
        num=handcards[k]&0x0F
        # 万牌 ab 【0-7】
        if handcards[k] & 0xF0 == 0x00 and handcards[k] + 1 in handcards:
            feature[num-1]+=1
        # 条 ab 【8-15】
        elif handcards[k] & 0xF0 == 0x10 and handcards[k] + 1 in handcards:
            feature[7+num]+=1
        # 筒 ab 【16-23】
        elif handcards[k] & 0xF0 == 0x20 and handcards[k] + 1 in handcards:
            feature[15+num]+=1
        k+=1
    #print feature
    return feature
#68-88
def ac(handcards):
    feature=[
         0,
         0,
         0,
         0,
         0,
         0,
         0,
         0,
         0,
         0,
         0,
         0,
         0,
         0,
         0,
         0,
         0,
         0,
         0,
         0,
         0,
    ]
    k = 0
    while k <= len(handcards) - 2:
        num = handcards[k] & 0x0F
        # 万 ac【0-6】
        if handcards[k] & 0xF0 == 0x00 and handcards[k] + 2 in handcards:
            feature[num-1] += 1
        # 条 【7-13】

        elif handcards[k] & 0xF0 == 0x10 and handcards[k] + 2 in handcards:
            #print handcards
            feature[6+ num] += 1
        # 筒 【14-20】
        elif handcards[k] & 0xF0 == 0x20 and handcards[k] + 2 in handcards:
            feature[13+ num] += 1
        k += 1
    #print  feature
    return feature

#89-109
def abc(handcards):
    feature=[
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    k = 0
    # print feature
    while k <= len(handcards) - 3:
        num = handcards[k] & 0x0F
        # wan abc [0-6]
        if handcards[k] & 0xF0 == 0x00 and handcards[k] + 2 in handcards and handcards[k] + 1 in handcards:
            feature[num-1] += 1
        # tiao abc [7-13]
        elif handcards[k] & 0xF0 == 0x10 and handcards[k] + 2 in handcards and handcards[k] + 1 in handcards:
            feature[6 + num] += 1
        # tong abc [14-20]
        elif handcards[k] & 0xF0 == 0x20 and handcards[k] + 2 in handcards and handcards[k] + 1 in handcards:
            feature[13 + num] += 1
        k += 1

    #print feature
    return feature



# 110
def kindOfKing(kingcard):
    color = (kingcard & 0xF0)/16
    num = kingcard & 0x0F
    result=0
    if color==0:
        result=num
    elif color==1:
        result=9+num
    elif color==2:
        result=18+num
    elif color==3:
        result=27+num
    #print  result
    return result
# 111
def num_king(kingList):
    num=len(kingList)
    #print  num
    return num
# 112-115
def fun_king(kingList):
    king=[0,0,0,0]
    i=0
    for kingcard in kingList:
        king[i]=kingcard
        i=i+1
    #print king
    return king
#116 飞宝数
#117-129
def fun_19(handCards):
    num = numofcard(handCards)
    num_19 = [num[0],num[8],num[9],num[17],num[18]]
    num_19 = num_19+num[26:]
    return num_19

def cal_13lan_funciton(tiles, result):
    num = len(tiles)
    #print('num=%d' % num)
    # if num == 2:
    #     # pre precess
    #     if tiles[0] not in (1, 6) or tiles[1] - tiles[0] < 3:
    #         return result
    #     if tiles[0] == 1:
    #         result[tiles[1] - 4] = 1
    #     if tiles[0] == 2:
    #         result[tiles[1] + 1] = 1
    #     if tiles[0] == 3:
    #         result[tiles[1] + 5] = 1
    #     if tiles[0] == 4:
    #         result[tiles[1] + 8] = 1
    #     if tiles[0] == 5:
    #         result[tiles[1] + 10] = 1
    #     if tiles[0] == 6:
    #         result[tiles[1] + 11] = 1
    for i in range(num - 1):
        for j in range(i + 1, num):
            if tiles[i] not in range(1, 6) or tiles[j] - tiles[i] < 3:
                #print(i, j)
                continue
            if tiles[i] == 1:
                result[tiles[j] - 4] = 1
            if tiles[i] == 2:
                result[tiles[j] + 1] = 1
            if tiles[i] == 3:
                result[tiles[j] + 5] = 1
            if tiles[i] == 4:
                result[tiles[j] + 8] = 1
            if tiles[i] == 5:
                result[tiles[j] + 10] = 1
            if tiles[i] == 6:
                result[tiles[j] + 11] = 1

    if num >= 3:
        if (tiles.count(1) + tiles.count(4) + tiles.count(7)) == 3:
            result[0] = result[3] = result[15] = result[21] = 1

        if (tiles.count(1) + tiles.count(4) + tiles.count(8)) == 3:
            result[0] = result[4] = result[16] = result[22] = 1

        if (tiles.count(1) + tiles.count(4) + tiles.count(9)) == 3:
            result[0] = result[5] = result[17] = result[23] = 1

        if (tiles.count(1) + tiles.count(5) + tiles.count(8)) == 3:
            result[1] = result[4] = result[18] = result[24] = 1

        if (tiles.count(1) + tiles.count(5) + tiles.count(9)) == 3:
            result[1] = result[5] = result[19] = result[25] = 1

        if (tiles.count(1) + tiles.count(6) + tiles.count(9)) == 3:
            result[2] = result[5] = result[20] = result[26] = 1

        if (tiles.count(2) + tiles.count(5) + tiles.count(8)) == 3:
            result[6] = result[9] = result[18] = result[27] = 1

        if (tiles.count(2) + tiles.count(5) + tiles.count(9)) == 3:
            result[6] = result[10] = result[19] = result[28] = 1

        if (tiles.count(2) + tiles.count(6) + tiles.count(9)) == 3:
            result[7] = result[10] = result[20] = result[29] = 1

        if (tiles.count(3) + tiles.count(6) + tiles.count(9)) == 3:
            result[11] = result[14] = result[20] = result[30] = 1

    return result


# 130-222 十三烂 2张的可能 真值
def cal_13lan_2tiles(tile_list):
    tmp = tile_list
    result1 = [0] * 31
    result2 = [0] * 31
    result3 = [0] * 31
    L = set(tmp)  # 去除重复手牌
    L_num0 = []  # 万数牌
    L_num1 = []  # 条数牌
    L_num2 = []  # 筒数牌
    for i in L:
        if i & 0xf0 == 0x00:
            L_num0.append(i & 0x0f)
        if i & 0xf0 == 0x10:
            L_num1.append(i & 0x0f)
        if i & 0xf0 == 0x20:
            L_num2.append(i & 0x0f)

    # 万字 烂牌 14-19,25-29,36-39,47-49,58-59,69
    result1 = cal_13lan_funciton(L_num0, result1)
    #print(L_num0)
    # 条  烂牌 14-19,25-29,36-39,47-49,58-59,69
    result2 = cal_13lan_funciton(L_num1, result2)
    #print(L_num1)
    # 筒字 烂牌 14-19,25-29,36-39,47-49,58-59,69
    result3 = cal_13lan_funciton(L_num2, result3)
    #print(L_num2)
    return result1 + result2 + result3



def calculate1(handCards):
    feature = []

    # 1 通常手
    hand_cards = copy.deepcopy(handCards)
    f1 = wait_types_comm(hand_cards)
    feature.append(14 - f1)

    # 2 七对
    hand_cards = copy.deepcopy((handCards))
    f2 = wait_types_7(hand_cards)
    feature.append(14 - f2)

    # 3 十三浪
    hand_cards = copy.deepcopy((handCards))
    f3 = wait_types_13(hand_cards)
    feature.append(14 - f3)

    # 4 91
    hand_cards = copy.deepcopy((handCards))
    f4 = wait_types_19(hand_cards)
    feature.append(14 - f4)

    # 5-9 花色数
    hand_cards = copy.deepcopy((handCards))
    f5_9 = numOfCards(hand_cards)
    feature = feature + f5_9

    # 10-43 每张牌的数
    hand_cards = copy.deepcopy((handCards))
    f10_43 = numofcard(hand_cards)
    feature = feature + f10_43

    # 44-67 ab
    hand_cards = copy.deepcopy((handCards))
    f44_67 = ab(hand_cards)
    feature = feature + f44_67

    # 68_88 ac
    hand_cards = copy.deepcopy((handCards))
    f68_88 = ac(hand_cards)
    feature = feature + f68_88

    # 89-109 abc
    hand_cards = copy.deepcopy((handCards))
    f89_109 = abc(hand_cards)
    feature = feature + f89_109

    # 114-206 十三浪牌数
    hand_cards = copy.deepcopy((handCards))
    f114_206 = cal_13lan_2tiles(hand_cards)
    feature = feature + f114_206

    return feature

def calculate2(handCards,king_card,king_num,fei_king,fei):
    feature = []

    #1 通常手
    hand_cards = copy.deepcopy(handCards)
    f1=wait_types_comm(hand_cards)
    feature.append(14-f1)

    #2 七对
    hand_cards = copy.deepcopy((handCards))
    f2=wait_types_7(hand_cards)
    feature.append(14-f2)

    #3 十三浪
    hand_cards = copy.deepcopy((handCards))
    f3 = wait_types_13(hand_cards)
    feature.append(14-f3)

    #4 91
    hand_cards = copy.deepcopy((handCards))
    f4 = wait_types_19(hand_cards)
    feature.append(14-f4)

    #5-9 花色数
    hand_cards = copy.deepcopy((handCards))
    f5_9 = numOfCards(hand_cards)
    feature=feature+f5_9


    #10-43 每张牌的数
    hand_cards = copy.deepcopy((handCards))
    f10_43 = numofcard(hand_cards)
    feature=feature+f10_43

    #44-67 ab
    hand_cards = copy.deepcopy((handCards))
    f44_67 = ab(hand_cards)
    feature=feature+f44_67

    #68_88 ac
    hand_cards = copy.deepcopy((handCards))
    f68_88 = ac(hand_cards)
    feature=feature+f68_88

    #89-109 abc
    hand_cards = copy.deepcopy((handCards))
    f89_109 = abc(hand_cards)
    feature=feature+f89_109

    # 110 kingcard
    f110 = kindOfKing(king_card)
    feature.append(f110)

    # 111 num_kingcards
    feature.append(king_num)


    # 111 飞宝数
    feature.append(fei_king)

    #112 本手是否飞宝
    feature.append(fei)



    # 114-206 十三浪牌数
    hand_cards = copy.deepcopy((handCards))
    f114_206 = cal_13lan_2tiles(hand_cards)
    feature = feature + f114_206

    return feature

