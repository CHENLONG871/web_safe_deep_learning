#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：web_safe_deep_learning 
@File ：review_detect_2.py
@IDE  ：PyCharm 
@Author ：chenlong871
@Date ：2021/12/1 22:43 
'''

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn import svm
# from sklearn.feature_extraction.text import TfidfTransformer
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
from sklearn.ensemble import RandomForestClassifier

max_features = 200
max_document_length = 500
vocabulary = None
word2ver_bin = "review_word2vec.bin"
doc2ver_bin = "review_doc2vec.bin"
word2ver_bin_cnn1d = "review_word2vec_cnn1d.bin"


def do_svm_doc2vec(x_train, x_test, y_train, y_test):
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
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))


def do_nb_doc2vec(x_train, x_test, y_train, y_test):
    """
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return: 贝叶斯模型
    """
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))


def do_rf_doc2vec(x_train, x_test, y_train, y_test):
    """
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return: 定义随机森林模型
    """
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))


def do_cnn_doc2vec_2d(trainX, testX, trainY, testY):
    """
    :param trainX:
    :param testX:
    :param trainY:
    :param testY:
    :return: 定义二维卷积神经网络用于docvec,这里有对样本进行聚合的操作
    """
    trainX = trainX.reshape([-1, max_features, max_document_length, 1])
    testX = testX.reshape([-1, max_features, max_document_length, 1])
    # 对y处理得这段代码自己添加的。
    y_train_one_hot = []
    y_test_one_hot = []
    for i in range(len(trainY)):
        if ((i + 1) % 500 == 0 and trainY[i] == 0):
            y_train_one_hot.append([float(1), float(0)])
        elif ((i + 1) % 500 == 0 and trainY[i] == 1):
            y_train_one_hot.append([float(0), float(1)])
    for i in range(len(testY)):
        if ((i + 1) % 500 == 0 and testY[i] == 0):
            y_test_one_hot.append([float(1), float(0)])
        elif ((i + 1) % 500 == 0 and testY[i] == 1):
            y_test_one_hot.append([float(0), float(1)])
    trainY = np.array(y_train_one_hot)  #
    testY = np.array(y_test_one_hot)
    # Building convolutional network
    network = input_data(shape=[None, max_features, max_document_length, 1], name='input')
    network = conv_2d(network, 16, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = fully_connected(network, 128, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 256, activation='tanh')
    network = dropout(network, 0.8)
    # network = fully_connected(network, 10, activation='softmax')
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.01,
                         loss='categorical_crossentropy', name='target')
    # Training
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit({'input': trainX}, {'target': trainY}, n_epoch=20,
              validation_set=({'input': testX}, {'target': testY}),
              snapshot_step=100, show_metric=True, run_id='review')


def do_cnn_doc2vec(trainX, testX, trainY, testY):
    """
    :param trainX:
    :param testX:
    :param trainY:
    :param testY:
    :return: 定义一维卷积神经网络
    """
    global max_features
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)
    # Building convolutional network
    network = input_data(shape=[None, max_features], name='input')  # 注意这里的维度是一维向量
    network = tflearn.embedding(network, input_dim=1000000, output_dim=128, validate_indices=False)
    branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2")
    branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer="L2")
    branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer="L2")
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
              n_epoch=5, shuffle=True, validation_set=(testX, testY),
              show_metric=True, batch_size=100, run_id="review")


def do_dnn_doc2vec(x_train, x_test, y_train, y_test):
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
                        hidden_layer_sizes=(5, 2),
                        random_state=1)
    print(clf)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))
