#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：web_safe_deep_learning 
@File ：get_feature_word2vec_dga.py
@IDE  ：PyCharm 
@Author ：chenlong871
@Date ：2022/2/21 17:44 
'''

import datetime
from gensim.models import Doc2Vec
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np
import gensim
import multiprocessing
from sklearn.preprocessing import scale

# 给定超参数，下面的bin文件会保存在E:\pycharm_project\deep_learning_web_safe\dga\code
word2ver_bin = "dga_word2vec.bin"
word2ver3_bin = "dgaerror_word2vec.bin"
max_features = 40
dga_file = "E:/pycharm_project/deep_learning_web_safe/dga/data/dga.txt"
alexa_file = "E:/pycharm_project/deep_learning_web_safe/dga/data/dga1m.csv"


def load_alexa():
    """
    :return: 获取正样本
    """
    x = []
    data = pd.read_csv(alexa_file, sep=",", header=None)
    x = [i[1] for i in data.values]
    return x


# 负样本
def load_dga():
    """
    :return: 获取负样本
    """
    x = []
    data = pd.read_csv(dga_file, sep="\t", header=None)
    x = [i[1] for i in data.values]
    return x


"""
第四步：获取word2vec/doc2vec文本特征的预处理工作
"""


def cleanText(corpus):
    """
    :param corpus:
    :return: 对特殊符号与标点符号做处理
    """
    punctuation = """.,?!:;(){}[]"""
    corpus = [z.lower().replace('\n', '') for z in corpus]
    corpus = [z.replace('<br />', ' ') for z in corpus]

    # treat punctuation as individual words
    for c in punctuation:
        corpus = [z.replace(c, ' %s ' % c) for z in corpus]
    corpus = [z.split() for z in corpus]
    return corpus


def normalize_text(text):
    """
    :param text:
    :return: onvert text to lower-case and strip punctuation/symbols from words
    """
    norm_text = text.lower()

    # Replace breaks with spaces
    norm_text = norm_text.replace('<br />', ' ')

    # Pad punctuation with spaces on both sides 在标点符号的两侧加空格
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        norm_text = norm_text.replace(char, ' ' + char + ' ')

    return norm_text


def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus]
    return np.array(np.concatenate(vecs), dtype='float')


def buildWordVector(imdb_w2v, text, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count  # 这里取的是所有word2vec的均值用来计算
    return vec


# word2vec文本特征模型会将每一个词语转化为n维向量，所以内存会不够
def get_features_by_word2vec_dga():
    """
    :return: 获取word2vec文本特征
    """
    global max_features
    global word2ver_bin
    alexa = load_alexa()
    dga = load_dga()
    x = alexa + dga
    x_change = []
    # 这里我们把单词拆分成字母
    for i in range(len(x)):
        b = " ".join(x[i])
        x_change.append(b)
    # max_features=100
    y = [0] * len(alexa) + [1] * len(dga)
    x_train, x_test, y_train, y_test = train_test_split(x_change, y, test_size=0.4)

    x_train = cleanText(x_train)
    x_test = cleanText(x_test)

    x = x_train + x_test
    cores = multiprocessing.cpu_count()  # 获取当前计算机的cpu个数

    if os.path.exists(word2ver3_bin):  # 构建一个永远不存在的bin文件
        print("Find cache file %s" % word2ver_bin)
        model = gensim.models.Word2Vec.load(word2ver_bin)
    else:
        model = gensim.models.Word2Vec(size=max_features, window=10, min_count=1, iter=60, workers=cores)

        model.build_vocab(x)

        model.train(x, total_examples=model.corpus_count, epochs=model.iter)
        model.save(word2ver_bin)

    x_train = np.concatenate([buildWordVector(model, z, max_features) for z in x_train])
    x_train = scale(x_train)
    x_test = np.concatenate([buildWordVector(model, z, max_features) for z in x_test])
    x_test = scale(x_test)

    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    start_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    x_train, x_test, y_train, y_test = get_features_by_word2vec_dga()
    print("get_features_by_word2vec_dga:", x_train.shape, x_train[1, 2:10])
    end_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    print(start_time, end_time)
