# -*- coding: utf-8 -*-

import numpy as np
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
import codecs

from bert4keras.bert import load_pretrained_model
from bert4keras.utils import Tokenizer, load_vocab
from bert4keras.train import PiecewiseLinearLearningRate


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                x1, x2 = tokenizer.encode(text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield X1, Y
                    [X1, X2, Y] = [], [], []


if __name__ == "__main__":
    config_path = './chinese_L-12_H-768_A-12/bert_config.json'
    checkpoint_path = './chinese_L-12_H-768_A-12/bert_model.ckpt'
    dict_path = './chinese_L-12_H-768_A-12/vocab.txt'

    token_dict = load_vocab(dict_path) # 读取词典
    tokenizer = Tokenizer(token_dict) # 建立分词器
    print(tokenizer.tokenize(u'今天天气不错'))

    model = load_pretrained_model(config_path, checkpoint_path)  # 建立模型，加载权重
    output = Lambda(lambda x: x[:, 0])(model.output)
    output = Dense(1, activation='sigmoid')(output)
    model = Model(model.input, output)

    model.compile(
        loss='binary_crossentropy',
        # optimizer=Adam(1e-5),  # 用足够小的学习率
        optimizer=PiecewiseLinearLearningRate(Adam(1e-4), {1000: 1, 2000: 0.1}),
        metrics=['accuracy']
    )
    model.summary()

    import pandas as pd

    df_train = pd.read_csv("./data/THUCNews/model/train.txt", names=["content", "label"], sep="\t", nrows=1000)
    df_valid = pd.read_csv("./data/THUCNews/model/valid.txt", names=["content", "label"], sep="\t", nrows=100)

    train_data = list(df_train.itertuples(index=False, name=None))
    valid_data = list(df_valid.itertuples(index=False, name=None))

    train_D = data_generator(train_data)
    valid_D = data_generator(valid_data)

    maxlen = 32

    # model.fit_generator(
    #     train_D.__iter__(),
    #     steps_per_epoch=len(train_D),
    #     epochs=1,
    #     validation_data=valid_D.__iter__(),
    #     validation_steps=len(valid_D)
    # )



    # import shap
    #
    # # we use the first 100 training examples as our background dataset to integrate over
    # backgroud_sample = data_generator(train_data, batch_size=100)
    # for i in backgroud_sample:
    #     backgroud_sample = i
    #     break
    # print("back done.")
    # explainer = shap.DeepExplainer(model, backgroud_sample[0][:10])
    # print("shap done.")
    # # explain the first 10 predictions
    # # explaining each prediction requires 2 * background dataset size runs
    # shap_values = explainer.shap_values(backgroud_sample[0][:2])
    # print("shap value done.")


