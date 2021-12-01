#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：web_safe_deep_learning 
@File ：get_feature_word2vec_review.py
@IDE  ：PyCharm 
@Author ：chenlong871
@Date ：2021/12/1 23:06 
'''
# 'E:\\github_project\\web_safe_deep_learning'
from web_safe_deep_learning.util.root_config_review import load_all_files
import datetime

# 保存模型文件
word2ver_bin = "review_word2vec.bin"
doc2ver_bin = "review_doc2vec.bin"

print('start_time: ' + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))


# word2vec文本特征模型会将每一个词语转化为n维向量，所以在自己电脑上内存会不够
def get_features_by_word2vec():
    """
    :return:获取word2vec文本特征,最后对取值进行标准化处理
    """
    global max_features
    global word2ver_bin
    x_train, x_test, y_train, y_test = load_all_files()

    x_train = cleanText(x_train)
    x_test = cleanText(x_test)

    x = x_train + x_test
    cores = multiprocessing.cpu_count()  # 获取当前计算机的cpu个数

    if os.path.exists(word2ver_bin):
        print("Find cache file %s" % word2ver_bin)
        model = gensim.models.Word2Vec.load(word2ver_bin)
    else:
        model = gensim.models.Word2Vec(size=max_features, window=10, min_count=1, iter=60, workers=cores)

        model.build_vocab(x)

        model.train(x, total_examples=model.corpus_count, epochs=model.iter)
        model.save(word2ver_bin)

    x_train = np.concatenate([buildWordVector(model, z, max_features) for z in x_train])
    x_train = scale(x_train)
    x_test = np.concatenate([buildWordVector(model, z, max_features) for z in x_test])
    x_test = scale(x_test)

    return x_train, x_test, y_train, y_test


def get_features_by_word2vec_cnn_1d():
    """
    :return:获取word2vec文本特征,最后对取值进行归一化处理(CNN模型中输入值不能为负？)
    """
    global max_features
    global word2ver_bin_cnn1d
    x_train, x_test, y_train, y_test = load_all_files()

    x_train = cleanText(x_train)
    x_test = cleanText(x_test)

    x = x_train + x_test
    cores = multiprocessing.cpu_count()  # 获取当前计算机的cpu个数

    if os.path.exists(word2ver_bin_cnn1d):
        print("Find cache file %s" % word2ver_bin_cnn1d)
        model = gensim.models.Word2Vec.load(word2ver_bin_cnn1d)
    else:
        model = gensim.models.Word2Vec(size=max_features, window=10, min_count=1, iter=60, workers=cores)

        model.build_vocab(x)

        model.train(x, total_examples=model.corpus_count, epochs=model.iter)
        model.save(word2ver_bin_cnn1d)

    min_max_scaler = preprocessing.MinMaxScaler()
    x_train = np.concatenate([buildWordVector(model, z, max_features) for z in x_train])
    # x_train = scale(x_train) #这里标准化处理数据介于[-1,1],但是CNN模型是不能负数？
    x_train = min_max_scaler.fit_transform(x_train)
    x_test = np.concatenate([buildWordVector(model, z, max_features) for z in x_test])
    # x_test = scale(x_test)
    x_test = min_max_scaler.transform(x_test)

    return x_train, x_test, y_train, y_test


def get_features_by_doc2vec():
    """
    :return:获取doc2vec文本特征
    """
    global max_features
    x_train, x_test, y_train, y_test = load_all_files()

    x_train = cleanText(x_train)
    x_test = cleanText(x_test)

    # 这里的x_train作为输入不仅包括review，还包括段落id
    x_train = labelizeReviews(x_train, 'TRAIN')
    x_test = labelizeReviews(x_test, 'TEST')

    x = x_train + x_test
    cores = multiprocessing.cpu_count()
    # models = [
    # PV-DBOW
    #    Doc2Vec(dm=0, dbow_words=1, size=200, window=8, min_count=19, iter=10, workers=cores),
    # PV-DM w/average
    #    Doc2Vec(dm=1, dm_mean=1, size=200, window=8, min_count=19, iter=10, workers=cores),
    # ]
    if os.path.exists(doc2ver_bin):
        print("Find cache file %s" % doc2ver_bin)
        model = Doc2Vec.load(doc2ver_bin)
    else:
        model = Doc2Vec(dm=0, size=max_features, negative=5, hs=0, min_count=2, workers=cores, iter=60)
        model.build_vocab(x)
        model.train(x, total_examples=model.corpus_count, epochs=model.iter)
        model.save(doc2ver_bin)

    x_test = getVecs(model, x_test, max_features)
    x_train = getVecs(model, x_train, max_features)
    min_max_scaler = preprocessing.MinMaxScaler()  # 这里我们自己进行归一化处理，看看数据有没有用

    x_train = min_max_scaler.fit_transform(x_train)
    x_test = min_max_scaler.transform(x_test)

    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    start_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    x_train, x_test, y_train, y_test = get_features_by_word2vec()
    print("et_features_by_word2vec:", x_train.shape, x_train[1, 2:10])
    x_train, x_test_2, y_train_2, y_test_2 = get_features_by_word2vec_cnn_1d()
    print("get_features_by_word2vec_cnn_1d:", x_train.shape, x_train[1, 2:10])
    x_train, x_test_3, y_train_3, y_test_3 = get_features_by_doc2vec()
    print('get_features_by_doc2vec:', x_train.shape, x_train[1, 2:10])
    end_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    print(start_time, end_time)
