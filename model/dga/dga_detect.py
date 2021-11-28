#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：web_safe_deep_learning 
@File ：dga_detect.py
@IDE  ：PyCharm 
@Author ：chenlong871
@Date ：2021/11/25 22:33 
'''
"""
需要现在site-packages加入.pth文件
"""

#1导入相关的包
import web_safe_deep_learning.util.root_config  as rc
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences
from sklearn.neural_network import MLPClassifier
from tflearn.layers.normalization import local_response_normalization
#from tensorflow.contrib import learn
import gensim
import re
from collections import namedtuple
from random import shuffle
import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import classification_report
import xgboost as xgb
import lightgbm as lgb
from sklearn import preprocessing
from hmmlearn import hmm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell


def do_nb(x_train, x_test, y_train, y_test):
    """
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return: 定义贝叶斯模型
    """
    gnb = GaussianNB()
    gnb.fit(x_train,y_train)
    y_pred=gnb.predict(x_test)
    print(classification_report(y_test, y_pred))
    print (metrics.confusion_matrix(y_test, y_pred))


def do_xgboost(x_train, x_test, y_train, y_test):
    """
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return: 定义xgboost模型
    """
    xgb_model = xgb.XGBClassifier().fit(x_train, y_train)
    y_pred = xgb_model.predict(x_test)
    print(classification_report(y_test, y_pred))
    print (metrics.confusion_matrix(y_test, y_pred))


# 调参主要是体现在num_class,以及metri两个参数取值的差异上。
def do_lightgbm(x_train, x_test, y_train, y_test):
    """
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return: 定义lightgbm模型
    """
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_test = lgb.Dataset(x_test, y_test, reference=lgb_train)
    gbm_parm = {"nthread": 5,
                "num_class": 1,
                "objective": 'regression',
                "learning_rate": 0.01,
                "num_leaves": 64,
                "bagging_fraction": 0.9,
                "feature_fraction": 0.7,
                "lambda_l1": 0.001,
                "lambda_l2": 0,
                "bagging_seed": 200,
                'metric': 'rmse'
                }
    evals_result = {}

    gbm = lgb.train(gbm_parm, lgb_train,
                    num_boost_round=20000,
                    valid_sets=[lgb_train, lgb_test],
                    early_stopping_rounds=100,  # 如果超过100次效率没有得到提升
                    evals_result=evals_result,
                    verbose_eval=10)

    y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)
    y_pred = [1 if i > 0.5 else 0 for i in y_pred]
    print(classification_report(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))


def do_mlp(x_train, x_test, y_train, y_test):
    """
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return: 定义多层感知机模型
    """
    global max_features
    # Building deep neural network
    clf = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes = (5, 2),
                        random_state = 1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred))
    print (metrics.confusion_matrix(y_test, y_pred))


def do_rnn(trainX, testX, trainY, testY):
    """
    :param trainX:
    :param testX:
    :param trainY:
    :param testY:
    :return: 定义一层lstm，循环神经网络模型
    """
    max_document_length=32
    y_test=testY
    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)  #不到最大长度的数据填充0
    testX = pad_sequences(testX, maxlen=max_document_length, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)       #把标记数据进行二值化处理
    testY = to_categorical(testY, nb_classes=2)

    # Network building
    net = tflearn.input_data([None, max_document_length])  #定义RNN结构，使用最简单的单层LSTM结构
    net = tflearn.embedding(net, input_dim=10240000, output_dim=32)  #看看试着把维度降低效率是否会好点
    net = tflearn.lstm(net, 32, dropout=0.1)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                             loss='categorical_crossentropy')

    # Training
    model = tflearn.DNN(net, tensorboard_verbose=0,tensorboard_dir="dga_log")
    model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
              batch_size=10,run_id="dga",n_epoch=1)

    y_predict_list = model.predict(testX)
    #print y_predict_list

    y_predict = []
    for i in y_predict_list:
        print  (i[0])
        if i[0] > 0.5:
            y_predict.append(0)
        else:
            y_predict.append(1)

    print(classification_report(y_test, y_predict))
    print (metrics.confusion_matrix(y_test, y_predict))

