#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：web_safe_deep_learning 
@File ：train_review_real_data_2.py
@IDE  ：PyCharm 
@Author ：chenlong871
@Date ：2021/12/2 9:36 
'''
from web_safe_deep_learning.process.get_feature_word2vec_review import get_features_by_word2vec, \
    get_features_by_word2vec_cnn_1d, get_features_by_doc2vec
from web_safe_deep_learning.model.review.review_detect_2 import do_svm_doc2vec, do_rf_doc2vec, do_nb_doc2vec, \
    do_dnn_doc2vec, do_cnn_doc2vec, do_cnn_doc2vec_2d
import datetime

max_features = 200
max_document_length = 500
vocabulary = None
word2ver_bin = "review_word2vec.bin"
doc2ver_bin = "review_doc2vec.bin"
word2ver_bin_cnn1d = "review_word2vec_cnn1d.bin"

if __name__ == "__main__":
    """
    基于word2vec/doc2vec构建的文本特征，运行机器学习/深度学习的模型
    """
    start_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    print("start get_features_by_word2vec--->")
    x_train, x_test, y_train, y_test = get_features_by_word2vec()
    print("start SVM")
    do_svm_doc2vec(x_train, x_test, y_train, y_test)
    print("start NB")
    do_nb_doc2vec(x_train, x_test, y_train, y_test)
    print("start RF")
    do_rf_doc2vec(x_train, x_test, y_train, y_test)
    print("start DNN")
    do_dnn_doc2vec(x_train, x_test, y_train, y_test)

    print("start get_features_by_word2vec_cnn_1d--->")
    x_train, x_test, y_train, y_test = get_features_by_word2vec_cnn_1d()
    print("start SVM")
    do_svm_doc2vec(x_train, x_test, y_train, y_test)
    print("start NB")
    do_nb_doc2vec(x_train, x_test, y_train, y_test)
    print("start RF")
    do_rf_doc2vec(x_train, x_test, y_train, y_test)
    print("start DNN")
    do_dnn_doc2vec(x_train, x_test, y_train, y_test)
    print("start word2vec_CNN_1d")
    do_cnn_doc2vec(x_train, x_test, y_train, y_test)

    print("start get_features_by_doc2vec--->")
    x_train, x_test, y_train, y_test = get_features_by_doc2vec()
    print("start SVM")
    do_svm_doc2vec(x_train, x_test, y_train, y_test)
    print("start NB")
    do_nb_doc2vec(x_train, x_test, y_train, y_test)
    print("start RF")
    do_rf_doc2vec(x_train, x_test, y_train, y_test)
    print("start DNN")
    do_dnn_doc2vec(x_train, x_test, y_train, y_test)
    print("start doc2vec_CNN_1")
    do_cnn_doc2vec(x_train, x_test, y_train, y_test)
    print("start doc2vec_CNN_2D")
    do_cnn_doc2vec_2d(x_train, x_test, y_train, y_test)
    print('end_time: ' + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
