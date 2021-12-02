#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：web_safe_deep_learning 
@File ：root_config_review.py
@IDE  ：PyCharm 
@Author ：chenlong871
@Date ：2021/11/28 22:34 
'''

# 导入相关的包
import datetime
import os
from collections import namedtuple
import numpy as np

SentimentDocument = namedtuple('SentimentDocument', 'words tags')


def load_one_file(filename):
    """
    :param filename:
    :return: 读取一个文件，保存为字符串
    """
    x = ""
    with open(filename, encoding='gb18030', errors='ignore') as f:
        for line in f:
            line = line.strip('\n')
            line = line.strip('\r')
            x += line
    f.close()
    return x


def load_files_from_dir(rootdir):
    """
    :param rootdir:
    :return: 遍历读取目录下的文件，以字符串的形式保存到列表
    """
    x = []
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            v = load_one_file(path)
            x.append(v)
    return x


def load_all_files():
    """
    :return:读取所有的样本，并且添加样本标签
    """
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    path = "E:/pycharm_project/deep_learning_web_safe/review/data/train/pos/"
    print("Load %s" % path)
    x_train = load_files_from_dir(path)
    y_train = [0] * len(x_train)
    path = "E:/pycharm_project/deep_learning_web_safe/review/data/train/neg/"
    print("Load %s" % path)
    tmp = load_files_from_dir(path)
    y_train += [1] * len(tmp)
    x_train += tmp
    path = "E:/pycharm_project/deep_learning_web_safe/review/data/test/pos/"
    print("Load %s" % path)
    x_test = load_files_from_dir(path)
    y_test = [0] * len(x_test)
    path = "E:/pycharm_project/deep_learning_web_safe/review/data/test/neg/"
    print("Load %s" % path)
    tmp = load_files_from_dir(path)
    y_test += [1] * len(tmp)
    x_test += tmp

    return x_train, x_test, y_train, y_test


# 对特殊符号与标点符号做处理
def cleanText(corpus):
    """
    :param corpus:
    :return: 去点特殊字符，以及把标点符号当作独立的文本
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
    :return: 对文本标准化处理
    """
    norm_text = text.lower()
    # Replace breaks with spaces
    norm_text = norm_text.replace('<br />', ' ')
    # Pad punctuation with spaces on both sides 在标点符号的两侧加空格
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        norm_text = norm_text.replace(char, ' ' + char + ' ')
    return norm_text


def labelizeReviews(reviews, label_type):
    """
    :param reviews:
    :param label_type:
    :return: doc2vec需要给每个reviewoc2Vec处理的每个英文段落，需要使用一个唯一的标识来标记
    """
    labelized = []
    for i, v in enumerate(reviews):
        label = '%s_%s' % (label_type, i)
        # labelized.append(LabeledSentence(v, [label]))
        # labelized.append(LabeledSentence(words=v,tags=label))
        labelized.append(SentimentDocument(v, [label]))
    return labelized


def getvecs(model, corpus, size):
    """
    :param model:
    :param corpus:
    :param size:
    :return: 基于训练好的docvec模型，获取每个单词对应的文本特征
    """
    vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus]
    return np.array(np.concatenate(vecs), dtype='float')


def buildWordVector(imdb_w2v, text, size):
    """
    :param imdb_w2v:
    :param text:
    :param size:
    :return:获取每个文档最后的特征
    """
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


if __name__ == "__main__":
    print('start_time: ' + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
    x_train, x_test, y_train, y_test = load_all_files()
    print(type(x_train), len(x_train))
    print(type(y_train), len(y_train))
    print('end_time: ' + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
