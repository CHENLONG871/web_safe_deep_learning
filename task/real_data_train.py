#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：web_safe_deep_learning 
@File ：real_data_train.py
@IDE  ：PyCharm 
@Author ：chenlong871
@Date ：2021/11/28 12:28 
'''

from web_safe_deep_learning.process.get_feature_nlp import get_feature_2gram,get_feature_234gram,get_feature_tfidf
from web_safe_deep_learning.model.dga.dga_detect import do_nb,do_mlp,do_xgboost,do_rnn,do_lightgbm
"""
1.1:   2gram模型
"""
x_train, x_test, y_train, y_test = get_feature_2gram()
print("start nb ----->")
do_nb(x_train, x_test, y_train, y_test)
print("start xgboost ----->")
do_xgboost(x_train, x_test, y_train, y_test)
print("start mlp ----->")
do_mlp(x_train, x_test, y_train, y_test)
print("start lightgbm ---->")
do_lightgbm(x_train, x_test, y_train, y_test)