# -*- coding: utf-8 -*-
import copy
import feature_extract_v4

def translate(op_card):  # 转换op_card函数
    if op_card >= 1 and op_card <= 9:
        op_card = op_card - 1
    elif op_card >= 17 and op_card <= 25:
        op_card = op_card - 8
    elif op_card >= 33 and op_card <= 41:
        op_card = op_card - 15
    elif op_card >= 49 and op_card <= 55:
        op_card = op_card - 22
    elif op_card == 255:
        op_card = 34
    return op_card

def translate2(i):  #转换十进制到cards
    if i>=10 and i<=18:
        i = i+ 7
    elif i>=19 and i<=27:
        i = i+14
    elif i>=28 and i<=34:
        i= i+21
    return i

def write(fo,data):
    king_card = data["king_cards"][0]
    handCards = data["init_cards"]
    fei_king = 0  # 飞宝数
    results = []  # 特征值
    for action in data["actions"]:
        if action["action_type"] == "G":  # 抓
            handCards.append(action["op_card"])
            handCards.sort()
            print (handCards) #检测是否合理
            king_num = 0
            handCards0 = copy.deepcopy(handCards)  # 得到去掉king_card后的手牌，为以后使用
            while 1:  # 手牌中的king_card 计数
                if king_card in handCards0:
                    handCards0.remove(king_card)
                    king_num = king_num + 1
                else:
                    break
            tr_king_cards = []  # king_card转换后的牌列表

            # 是否胡牌，这里一直设置为0
            hu = 0
            if action["action_type"] == "A":
                hu = 1

            if king_num == 0:
                results = feature_extract_v4.calculate(handCards, handCards0, king_card, tr_king_cards, fei_king,
                                                       hu)  # 特征计算函数
                for fea in results:
                    fo.write(str(fea) + " ")
                #fo.write(str(translate(action["op_card"])) + "\n")
                fo.write(str(-1) + "\n")#测试使用，摸到牌没有标签，标签需要计算

            elif king_num == 1:
                for i in range(1, 35):
                    king1 = translate2(i)
                    tr_king_cards.append(i)
                    handCards0.append(king1)
                    # print king1
                    handCards0 = sorted(handCards0)
                    # print king1
                    results = feature_extract_v4.calculate(handCards, handCards0, king_card, tr_king_cards, fei_king,
                                                           hu)
                    for fea in results:
                        fo.write(str(fea) + " ")
                    fo.write(str(translate(action["op_card"])) + "\n")

                    handCards0.remove(king1)
                    tr_king_cards.remove(i)

            elif king_num == 2:
                for i in range(1, 35):
                    king1 = translate2(i)
                    tr_king_cards.append(i)
                    handCards0.append(king1)
                    for j in range(1, 35):
                        king2 = translate2(j)
                        tr_king_cards.append(j)
                        handCards0.append(king2)
                        handCards0 = sorted(handCards0)
                        results = feature_extract_v4.calculate(handCards, handCards0, king_card, tr_king_cards,
                                                               fei_king, hu)
                        for fea in results:
                            fo.write(str(fea) + " ")
                        fo.write(str(translate(action["op_card"])) + "\n")

                        handCards0.remove(king2)
                        tr_king_cards.remove(j)
                    handCards0.remove(king1)
                    tr_king_cards.remove(i)

            elif king_num == 3:
                for i in range(1, 35):
                    king1 = translate2(i)
                    tr_king_cards.append(i)
                    handCards0.append(king1)
                    for j in range(1, 35):
                        king2 = translate2(j)
                        tr_king_cards.append(j)
                        handCards0.append(king2)
                        for k in range(1, 35):
                            king3 = translate2(k)
                            tr_king_cards.append(k)
                            handCards0.append(king3)
                            handCards0 = sorted(handCards0)
                            results = feature_extract_v4.calculate(handCards, handCards0, king_card, tr_king_cards,
                                                                   fei_king, hu)
                            for fea in results:
                                fo.write(str(fea) + " ")
                            fo.write(str(translate(action["op_card"])) + "\n")
                            handCards0.remove(king3)
                            tr_king_cards.remove(k)
                        handCards0.remove(king2)
                        tr_king_cards.remove(j)
                    handCards0.remove(king1)
                    tr_king_cards.remove(i)

            elif king_num == 4:
                for i in range(1, 35):
                    king1 = translate2(i)
                    tr_king_cards.append(i)
                    handCards0.append(king1)
                    for j in range(1, 35):
                        king2 = translate2(j)
                        tr_king_cards.append(j)
                        handCards0.append(king2)
                        for k in range(1, 35):
                            king3 = translate2(k)
                            tr_king_cards.append(k)
                            handCards0.append(king3)
                            for l in range(1, 35):
                                king4 = translate2(l)
                                tr_king_cards.append(l)
                                handCards0.append(king4)
                                handCards0 = sorted(handCards0)
                                results = feature_extract_v4.calculate(handCards, handCards0, king_card, tr_king_cards,
                                                                       fei_king, hu)
                                for fea in results:
                                    fo.write(str(fea) + " ")
                                fo.write(str(translate(action["op_card"])) + "\n")
                                handCards0.remove(king4)
                                tr_king_cards.remove(l)
                            handCards0.remove(king3)
                            tr_king_cards.remove(k)
                        handCards0.remove(king2)
                        tr_king_cards.remove(j)
                    handCards0.remove(king1)
                    tr_king_cards.remove(i)

        elif action["action_type"] == "d": #出牌
            if action["op_card"] == king_card:  # 计算飞宝数
                fei_king = fei_king + 1
            handCards.remove(action["op_card"])
        elif action["action_type"] == "A":#胡牌，保留
            break
        else:#出错打印
            print (action["action_type"])
