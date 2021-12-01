#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：web_safe_deep_learning 
@File ：get_feature_nlp_review.py
@IDE  ：PyCharm 
@Author ：chenlong871
@Date ：2021/11/28 23:34 
'''
import tflearn
import numpy as np
import datetime
from sklearn.feature_extraction.text import CountVectorizer
from web_safe_deep_learning.util.root_config_review import load_all_files
from sklearn.feature_extraction.text import TfidfTransformer

# 给定超参数的取值
max_features = 200
# max_document_length=500  #词汇表模型中需要用到,这个参数的取值会影响到cnn模型的正常运行(out of list)
max_document_length = 100
vocabulary = None


def get_features_by_wordbag():
    """
    :return: 特征提取：1，词袋模型
    """
    global max_features
    x_train, x_test, y_train, y_test = load_all_files()

    vectorizer = CountVectorizer(
        decode_error='ignore',
        strip_accents='ascii',
        max_features=max_features,
        stop_words='english',
        max_df=1.0,
        min_df=1)
    print(vectorizer)
    x_train = vectorizer.fit_transform(x_train)
    x_train = x_train.toarray()
    vocabulary = vectorizer.vocabulary_

    # 用训练集生成的词袋模型对测试数据进行词袋化处理
    vectorizer = CountVectorizer(
        decode_error='ignore',
        strip_accents='ascii',
        vocabulary=vocabulary,
        stop_words='english',
        max_df=1.0,
        min_df=1)
    print(vectorizer)
    x_test = vectorizer.fit_transform(x_test)
    x_test = x_test.toarray()

    return x_train, x_test, y_train, y_test


def get_features_by_wordbag_tfidf():
    """
    :return: 特征提取：2，TF-IDF模型
    """
    global max_features
    x_train, x_test, y_train, y_test = load_all_files()

    vectorizer = CountVectorizer(
        decode_error='ignore',
        strip_accents='ascii',
        max_features=max_features,
        stop_words='english',
        max_df=1.0,
        min_df=1,
        binary=True)
    print(vectorizer)
    x_train = vectorizer.fit_transform(x_train)
    x_train = x_train.toarray()
    vocabulary = vectorizer.vocabulary_
    # 用训练集生成的词袋模型对测试数据进行词袋化处理
    vectorizer = CountVectorizer(
        decode_error='ignore',
        strip_accents='ascii',
        vocabulary=vocabulary,
        stop_words='english',
        max_df=1.0, binary=True,
        min_df=1)
    print(vectorizer)
    x_test = vectorizer.fit_transform(x_test)
    x_test = x_test.toarray()

    transformer = TfidfTransformer(smooth_idf=False)
    x_train = transformer.fit_transform(x_train)
    x_train = x_train.toarray()
    x_test = transformer.transform(x_test)
    x_test = x_test.toarray()

    return x_train, x_test, y_train, y_test


def get_features_by_tf():
    """
    :return: 3:词汇表模型
    """
    global max_document_length
    x_train, x_test, y_train, y_test = load_all_files()

    vp = tflearn.data_utils.VocabularyProcessor(max_document_length=max_document_length,
                                                min_frequency=0,
                                                vocabulary=None,
                                                tokenizer_fn=None)
    x_train = vp.fit_transform(x_train, unused_y=None)
    x_train = np.array(list(x_train))

    x_test = vp.transform(x_test)
    x_test = np.array(list(x_test))
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    start_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    x_train, x_test_1, y_train_1, y_test_1 = get_features_by_wordbag()
    print('get_features_by_wordbag:', x_train.shape)
    x_train, x_test_2, y_train_2, y_test_2 = get_features_by_wordbag_tfidf()
    print('get_features_by_wordbag_tfidf:', x_train.shape)
    x_train, x_test_3, y_train_3, y_test_3 = get_features_by_tf()
    print('get_features_by_tf:', x_train.shape)
    end_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    print(start_time, end_time)
