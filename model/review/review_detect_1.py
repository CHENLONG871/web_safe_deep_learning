#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：web_safe_deep_learning 
@File ：review_detect_1.py
@IDE  ：PyCharm 
@Author ：chenlong871
@Date ：2021/11/28 23:43 
'''
#导入相关的包
import datetime


from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
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
from web_safe_deep_learning.process.get_feature_nlp_review import get_features_by_wordbag
from tflearn.layers.normalization import local_response_normalization
#from tensorflow.contrib import learn

#一些超参数的设置
max_features=200
max_document_length=500  #词汇表模型中需要用到,这个参数的取值会影响到cnn模型的正常运行(out of list)
vocabulary=None

#通过贝叶斯模型的效果来对特征的个数做决定。
def show_diffrent_max_features():
    """
    :return: 通过贝叶斯模型的效果来对特征的个数做决定。
    """
    global max_features
    a=[]
    b=[]
    for i in range(1000,20000,2000):
        max_features=i
        print ("max_features=%d" % i)
        #x, y = get_features_by_wordbag()
        #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
        x_train, x_test, y_train, y_test=get_features_by_wordbag()
        gnb = GaussianNB()
        gnb.fit(x_train, y_train)
        y_pred = gnb.predict(x_test)
        score=metrics.accuracy_score(y_test, y_pred)
        a.append(max_features)
        b.append(score)
        plt.plot(a, b, 'r')
    plt.xlabel("max_features")
    plt.ylabel("metrics.accuracy_score")
    plt.title("metrics.accuracy_score VS max_features")
    plt.legend()
    plt.show()


def do_nb_wordbag(x_train, x_test, y_train, y_test):
    """
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return: 贝叶斯模型
    """
    gnb = GaussianNB()
    gnb.fit(x_train,y_train)
    y_pred=gnb.predict(x_test)
    print (metrics.accuracy_score(y_test, y_pred))
    print (metrics.confusion_matrix(y_test, y_pred))

def do_svm_wordbag(x_train, x_test, y_train, y_test):
    """
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return: 支持向量机模型
    """
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print (metrics.accuracy_score(y_test, y_pred))
    print (metrics.confusion_matrix(y_test, y_pred))

def do_cnn_wordbag(trainX, testX, trainY, testY):
    """
    :param trainX:
    :param testX:
    :param trainY:
    :param testY:
    :return: 一维卷积神经网络
    """
    global max_document_length
    #print ("CNN and tf")

    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.) #对数据进行填充，不带最大长度的用0替换
    testX = pad_sequences(testX, maxlen=max_document_length, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Building convolutional network(change output_dim = 64 --> output_dim = 32,n_epoch= 5 --> n_epoch =1)
    network = input_data(shape=[None,max_document_length], name='input')
    network = tflearn.embedding(network, input_dim=10240000, output_dim=32)
    branch1 = conv_1d(network, 32, 3, padding='valid', activation='relu', regularizer="L2")
    branch2 = conv_1d(network, 32, 4, padding='valid', activation='relu', regularizer="L2")
    branch3 = conv_1d(network, 32, 5, padding='valid', activation='relu', regularizer="L2")
    network = merge([branch1, branch2, branch3], mode='concat', axis=1)
    network = tf.expand_dims(network, 2)
    network = global_max_pool(network)
    network = dropout(network, 0.8)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy', name='target')
    # Training
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit(trainX, trainY,
              n_epoch=1, shuffle=True, validation_set=(testX, testY),
              show_metric=True, batch_size=100,run_id="review")


def do_rnn_wordbag(trainX, testX, trainY, testY):
    """
    :param trainX:
    :param testX:
    :param trainY:
    :param testY:
    :return: lstm(RNN)长短期记忆循环神经网络
    """
    global max_document_length
    #print ("RNN and wordbag")

    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    testX = pad_sequences(testX, maxlen=max_document_length, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Network building
    net = tflearn.input_data([None, max_document_length])
    net = tflearn.embedding(net, input_dim=10240000, output_dim=32)
    net = tflearn.lstm(net, 32, dropout=0.8)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                             loss='categorical_crossentropy')

    # Training
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
              batch_size=10,run_id="review",n_epoch=1)


def do_dnn_wordbag(x_train, x_test, y_train, y_test):
    """
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return: 多层感知机模型
    """
    clf = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes = (5, 2),
                        random_state = 1)
    print  (clf)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print (metrics.accuracy_score(y_test, y_pred))
    print (metrics.confusion_matrix(y_test, y_pred))