#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：web_safe_deep_learning 
@File ：root_config.py
@IDE  ：PyCharm 
@Author ：chenlong871
@Date ：2021/11/25 23:19 
'''
import pandas as pd
import datetime

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
    print('start_time: ' + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
    dga = load_dga()
    alexa = load_alexa()
    print(type(dga), len(dga))
    print(type(alexa),len(alexa))
    print('send_time: ' + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
