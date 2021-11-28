#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：web_safe_deep_learning 
@File ：train_dga_real_data.py
@IDE  ：PyCharm 
@Author ：chenlong871
@Date ：2021/11/28 13:48 
'''

from web_safe_deep_learning.process.get_feature_nlp import get_feature_2gram, get_feature_234gram, get_feature_tfidf
from web_safe_deep_learning.process.get_feature_statistic import get_feature, get_feature_charseq
from web_safe_deep_learning.model.dga.dga_detect import do_nb, do_mlp, do_xgboost, do_rnn, do_lightgbm
import datetime

if __name__ == "__main__":
    """
    基于各种构建特征的方法，运行机器学习/深度学习的模型
    """
    start_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    print("get_feature_2gram")
    x_train, x_test, y_train, y_test = get_feature_2gram()
    print("start nb ----->")
    do_nb(x_train, x_test, y_train, y_test)
    print("start xgboost ----->")
    do_xgboost(x_train, x_test, y_train, y_test)
    print("start mlp ----->")
    do_mlp(x_train, x_test, y_train, y_test)
    print("start lightgbm ---->")
    do_lightgbm(x_train, x_test, y_train, y_test)

    print("get_feature_tfidf")
    x_train, x_test, y_train, y_test = get_feature_tfidf()
    print("start nb ----->")
    do_nb(x_train, x_test, y_train, y_test)
    print("start xgboost ----->")
    do_xgboost(x_train, x_test, y_train, y_test)
    print("start mlp ----->")
    do_mlp(x_train, x_test, y_train, y_test)
    print("start lightgbm ---->")
    do_lightgbm(x_train, x_test, y_train, y_test)

    print("get_feature_statistic")
    x_train, x_test, y_train, y_test = get_feature()
    print("start nb ----->")
    do_nb(x_train, x_test, y_train, y_test)
    print("start xgboost ----->")
    do_xgboost(x_train, x_test, y_train, y_test)
    print("start mlp ----->")
    do_mlp(x_train, x_test, y_train, y_test)
    print("start lightgbm ---->")
    do_lightgbm(x_train, x_test, y_train, y_test)

    print("get_feature_234gram")
    x_train, x_test, y_train, y_test = get_feature_234gram()
    print("start nb ----->")
    do_nb(x_train, x_test, y_train, y_test)
    print("start xgboost ----->")
    do_xgboost(x_train, x_test, y_train, y_test)
    print("start mlp ----->")
    do_mlp(x_train, x_test, y_train, y_test)
    print("start lightgbm ---->")
    do_lightgbm(x_train, x_test, y_train, y_test)

    print("get_feature_charseq")
    x_train, x_test, y_train, y_test = get_feature_charseq()
    do_rnn(x_train, x_test, y_train, y_test)
    end_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    print(start_time, end_time)
