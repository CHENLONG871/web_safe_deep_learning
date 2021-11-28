#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：web_safe_deep_learning 
@File ：root_config_review.py
@IDE  ：PyCharm 
@Author ：chenlong871
@Date ：2021/11/28 22:34 
'''

#导入相关的包
import datetime
import os

def load_one_file(filename):
    """
    :param filename:
    :return: 读取一个文件，保存为字符串
    """
    x=""
    with open(filename,encoding='gb18030', errors='ignore') as f:
        for line in f:
            line=line.strip('\n')
            line = line.strip('\r')
            x+=line
    f.close()
    return x

def load_files_from_dir(rootdir):
    """
    :param rootdir:
    :return: 遍历读取目录下的文件，以字符串的形式保存到列表
    """
    x=[]
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            v=load_one_file(path)
            x.append(v)
    return x

def load_all_files():
    """
    :return:读取所有的样本，并且添加样本标签
    """
    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]
    path="E:/pycharm_project/deep_learning_web_safe/review/data/train/pos/"
    print ("Load %s" % path)
    x_train=load_files_from_dir(path)
    y_train=[0]*len(x_train)
    path="E:/pycharm_project/deep_learning_web_safe/review/data/train/neg/"
    print ("Load %s" % path)
    tmp=load_files_from_dir(path)
    y_train+=[1]*len(tmp)
    x_train+=tmp

    path="E:/pycharm_project/deep_learning_web_safe/review/data/test/pos/"
    print ("Load %s" % path)
    x_test=load_files_from_dir(path)
    y_test=[0]*len(x_test)
    path="E:/pycharm_project/deep_learning_web_safe/review/data/test/neg/"
    print ("Load %s" % path)
    tmp=load_files_from_dir(path)
    y_test+=[1]*len(tmp)
    x_test+=tmp

    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    print('start_time: ' + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
    x_train, x_test, y_train, y_test = load_all_files()
    print(type(x_train),len(x_train))
    print(type(y_train), len(y_train))
    print('end_time: ' + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))


