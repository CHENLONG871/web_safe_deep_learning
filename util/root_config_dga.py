#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：web_safe_deep_learning 
@File ：root_config_dga.py
@IDE  ：PyCharm 
@Author ：chenlong871
@Date ：2021/11/28 22:35 
'''

import pandas as pd

dga_file = "E:/pycharm_project/deep_learning_web_safe/dga/data/dga.txt"
alexa_file = "E:/pycharm_project/deep_learning_web_safe/dga/data/dga1m.csv"


# 读取基本的数据
def load_alexa():
    """
    :return: 获取正常域名
    """
    # x = []
    data = pd.read_csv(alexa_file, sep=",", header=None)
    x = [i[1] for i in data.values]
    return x


# 负样本
def load_dga():
    """
    :return: 获取DGA域名
    """
    # x = []
    data = pd.read_csv(dga_file, sep="\t", header=None)
    x = [i[1] for i in data.values]
    return x


# 开始测试则会一部脚本
if __name__ == "__main__":
    dga = load_dga()
    alexa = load_alexa()
    print(type(dga), len(dga))
    print(type(alexa),len(alexa))
