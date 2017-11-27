# -*-coding:utf-8 -*-
# __author__='Yan'
# function: 计算v4版和十三烂相关的特征
import copy
import sys

reload(sys)
sys.setdefaultencoding('utf-8')


#  十三浪的相关特征计算


def cal_13lan_funciton(tiles, result):
    num = len(tiles)
    print('num=%d' % num)
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
                print(i, j)
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
            result[0] = result[5] = result[17] = result[22] = 1

        if (tiles.count(1) + tiles.count(5) + tiles.count(8)) == 3:
            result[1] = result[4] = result[18] = result[22] = 1

        if (tiles.count(1) + tiles.count(5) + tiles.count(9)) == 3:
            result[1] = result[5] = result[19] = result[22] = 1

        if (tiles.count(1) + tiles.count(6) + tiles.count(9)) == 3:
            result[2] = result[5] = result[20] = result[22] = 1

        if (tiles.count(2) + tiles.count(5) + tiles.count(8)) == 3:
            result[6] = result[9] = result[18] = result[27] = 1

        if (tiles.count(2) + tiles.count(5) + tiles.count(9)) == 3:
            result[6] = result[10] = result[19] = result[28] = 1

        if (tiles.count(2) + tiles.count(6) + tiles.count(9)) == 3:
            result[7] = result[10] = result[20] = result[29] = 1

        if (tiles.count(3) + tiles.count(6) + tiles.count(9)) == 3:
            result[11] = result[14] = result[20] = result[30] = 1

    return result


# 110-170 十三烂 2张的可能 真值
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
    print(L_num0)
    # 条  烂牌 14-19,25-29,36-39,47-49,58-59,69
    result2 = cal_13lan_funciton(L_num1, result2)
    print(L_num1)
    # 筒字 烂牌 14-19,25-29,36-39,47-49,58-59,69
    result3 = cal_13lan_funciton(L_num2, result3)
    print(L_num2)
    return result1 + result2 + result3


# 测试用例
hands_test = [0x02, 0x03, 0x03, 0x05, 0x06, 0x09, 0x03, 0x14, 0x16, 0x28, 0x32, 0x34, 0x36]

lll = cal_13lan_2tiles(hands_test)
test = []
# 打印特征位置
print(lll)
for i in range(len(lll)):
    if lll[i] == 1:
        test.append(i)
print(test)
