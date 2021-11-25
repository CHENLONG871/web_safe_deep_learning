#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：web_safe_deep_learning 
@File ：get_feature_nlp.py
@IDE  ：PyCharm 
@Author ：chenlong871
@Date ：2021/11/25 23:37 
'''
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from web_safe_deep_learning.util.root_config import load_dga, load_alexa
from sklearn.model_selection import train_test_split


def get_feature_2gram():
    """
    :return:通过2gram获取文本特征
    """
    alexa = load_alexa()
    dga = load_dga()
    x = alexa + dga
    max_features = 10000
    y = [0] * len(alexa) + [1] * len(dga)

    CV = CountVectorizer(
        ngram_range=(2, 2),
        token_pattern=r'\w',
        decode_error='ignore',
        strip_accents='ascii',
        max_features=max_features,
        stop_words='english',
        max_df=1.0,
        min_df=1)
    x = CV.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

    return x_train.toarray(), x_test.toarray(), y_train, y_test


def get_feature_tfidf():
    """
    :return:通过TF-IDF提取文本特征
    """
    alexa = load_alexa()
    dga = load_dga()
    x = alexa + dga
    max_features = 10000
    y = [0] * len(alexa) + [1] * len(dga)
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='ascii',
        token_pattern=r'\w',
        decode_error='ignore',
        stop_words='english',
        ngram_range=(1, 1),
        max_features=10000)
    word_vectorizer.fit(x)  # 这里确实是拿了全部的样本的
    train_word_features = word_vectorizer.transform(x)

    x_train, x_test, y_train, y_test = train_test_split(train_word_features, y, test_size=0.4)
    return x_train.toarray(), x_test.toarray(), y_train, y_test


def get_feature_234gram():
    """
    :return:通过234_gram获取文本特征
    """
    alexa = load_alexa()
    dga = load_dga()
    x = alexa + dga
    max_features = 10000
    y = [0] * len(alexa) + [1] * len(dga)

    CV = CountVectorizer(
        ngram_range=(2, 4),
        token_pattern=r'\w',
        decode_error='ignore',
        strip_accents='ascii',
        max_features=max_features,
        stop_words='english',
        max_df=1.0,
        min_df=1)
    x = CV.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

    return x_train.toarray(), x_test.toarray(), y_train, y_test


