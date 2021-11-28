#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：web_safe_deep_learning 
@File ：train_review_real_data_1.py
@IDE  ：PyCharm 
@Author ：chenlong871
@Date ：2021/11/29 0:02 
'''
from web_safe_deep_learning.process.get_feature_nlp_review import get_features_by_wordbag, get_features_by_tf,get_features_by_wordbag_tfidf
from web_safe_deep_learning.model.review.review_detect_1 import do_dnn_wordbag, do_nb_wordbag, do_cnn_wordbag, \
    do_rnn_wordbag, do_svm_wordbag, show_diffrent_max_features
import datetime

#给定超参数的个数
ax_features=200
max_document_length=500
vocabulary=None

if __name__ == "__main__":
    """
    基于各种构建特征的方法，运行机器学习/深度学习的模型
    """
    start_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    print("选定超参数的个数")
    show_diffrent_max_features()

    print("start get_features_by_wordbag --->")
    x_train, x_test, y_train, y_test = get_features_by_wordbag()
    # Nb
    print("start NB")
    do_nb_wordbag(x_train, x_test, y_train, y_test)
    # SVM
    print("start SVM")
    do_svm_wordbag(x_train, x_test, y_train, y_test)
    # 多层感知机。
    print("start DNN")
    do_dnn_wordbag(x_train, x_test, y_train, y_test)
    #一维卷积神经网络
    print("start CNN")
    do_cnn_wordbag(x_train, x_test, y_train, y_test)

    print("start get_features_by_wordbag_tfidf --->")
    x_train, x_test, y_train, y_test = get_features_by_wordbag_tfidf()
    # Nb
    print("start NB")
    do_nb_wordbag(x_train, x_test, y_train, y_test)
    # SVM
    print("start SVM")
    do_svm_wordbag(x_train, x_test, y_train, y_test)
   # 多层感知机。
    print("start DNN")  # 这里的结果很奇怪
    do_dnn_wordbag(x_train, x_test, y_train, y_test)
    # 卷积神经网络。
    print("start CNN")
    do_cnn_wordbag(x_train, x_test, y_train, y_test)

    print("start get_features_by_wordbag_tf --->")
    x_train, x_test, y_train, y_test = get_features_by_tf()
    # Nb
    print("start NB")
    do_nb_wordbag(x_train, x_test, y_train, y_test)
    # SVM
    print("start SVM")
    do_svm_wordbag(x_train, x_test, y_train, y_test)
    # 多层感知机。
    print("start DNN")  # 这里的结果很奇怪
    do_dnn_wordbag(x_train, x_test, y_train, y_test)
    # 卷积神经网络。
    print("start CNN")
    do_cnn_wordbag(x_train, x_test, y_train, y_test)
    #循环神经网络
    print("start RNN")
    do_rnn_wordbag(x_train, x_test, y_train, y_test)
    end_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    print(start_time,end_time)






