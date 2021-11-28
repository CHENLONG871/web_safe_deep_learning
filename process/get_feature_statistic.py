#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：web_safe_deep_learning
@File ：get_feature_statistic.py
@IDE  ：PyCharm 
@Author ：chenlong871
@Date ：2021/11/25 23:51 
'''
from web_safe_deep_learning.util.root_config import load_dga, load_alexa
from sklearn.model_selection import train_test_split
import re


def get_aeiou(domain):
    """
    :param domain:
    :return:返回元音字母的个数
    """
    count = len(re.findall(r'[aeiou]', domain.lower()))
    # count = (0.0 + count) / len(domain)
    return count


def get_uniq_char_num(domain):
    """
    :param domain:
    :return: 返回去重以后字母的个数
    """
    count = len(set(domain))
    # count=(0.0+count)/len(domain)
    return count


def get_uniq_num_num(domain):
    """
    :param domain:
    :return: 返回的是字母的个数
    """
    count = len(re.findall(r'[1234567890]', domain.lower()))
    # count = (0.0 + count) / len(domain)
    return count


def get_feature():
    from sklearn import preprocessing
    alexa = load_alexa()
    dga = load_dga()
    v = alexa + dga
    y = [0] * len(alexa) + [1] * len(dga)
    x = []

    for vv in v:
        vvv = [get_aeiou(vv), get_uniq_char_num(vv), get_uniq_num_num(vv), len(vv)]
        x.append(vvv)

    x = preprocessing.scale(x)  # 标准化处理
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
    return x_train, x_test, y_train, y_test


#当前的脚本没有问题
def get_feature_charseq():
    """
    :return:字符序列模型,字符组成的序列，字符转换成对应的ASCII值，这样就可以把域名最终转换成一个数字序列：可以理解为是具有时序特征的时序序列
    """
    alexa = load_alexa()
    dga = load_dga()
    x = alexa + dga
    max_features = 10000
    y = [0] * len(alexa) + [1] * len(dga)

    t = []
    for i in x:
        v = []
        for j in range(0, len(i)):
            v.append(ord(i[j]))
        t.append(v)

    x = t
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = get_feature()
    print('get_feature_statistic:', x_train.shape)
    x_train, x_test_3, y_train_3, y_test_3 = get_feature_charseq()
    print('get_feature_charseq:', type(x_train))
