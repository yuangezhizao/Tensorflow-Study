#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
    :Author: yuangezhizao
    :Time: 2019/3/23 0023 17:53
    :Site: https://www.yuangezhizao.cn
    :Copyright: © 2019 yuangezhizao <root@yuangezhizao.cn>
"""
# 0导入模块，生成模拟数据集
import numpy as np
import matplotlib.pyplot as plt

seed = 2


def generateds():
    # 基于 seed 产生随机数
    rdm = np.random.RandomState(seed)
    # 随机数返回 300 行 2 列的矩阵，表示 300 组坐标点（x0，x1）作为输入数据集
    X = rdm.randn(300, 2)
    # 从 X 这个 300 行 2 列的矩阵中取出一行，判断如果两个坐标的平方和小于 2，给 Y 赋值 1，其余赋值 0
    # 作为输入数据集的标签（正确答案）
    Y_ = [int(x0 * x0 + x1 * x1 < 2) for (x0, x1) in X]
    # 遍历 Y 中的每个元素，1 赋值 'red' 其余赋值 'blue'，这样可视化显示时人可以直观区分
    Y_c = [['red' if y else 'blue'] for y in Y_]
    # 对数据集 X 和标签 Y 进行形状整理，第一个元素为 -1 表示跟随第二列计算，第二个元素表示多少列，可见 X 为两列，Y 为 1 列
    X = np.vstack(X).reshape(-1, 2)
    Y_ = np.vstack(Y_).reshape(-1, 1)

    print(X)
    print(Y_)
    print(Y_c)
    # 用 plt.scatter 画出数据集 X 各行中第 0 列元素和第 1 列元素的点即各行的（x0，x1），用各行 Y_c 对应的值表示颜色（c 是 color 的缩写）
    plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
    plt.show()

    return X, Y_, Y_c
