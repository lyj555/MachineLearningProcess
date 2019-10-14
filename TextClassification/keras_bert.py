# -*- coding: utf-8 -*-

from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import codecs


config_path = './chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './chinese_L-12_H-768_A-12/vocab.txt'

token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


tokenizer = OurTokenizer(token_dict)
tokenizer.tokenize(u'今天天气不错')


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, batch_size=32, max_len=32):
        self.data = data
        self.batch_size = batch_size
        self.max_len = max_len
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = range(len(self.data))
            # np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:self.max_len]
                x1, x2 = tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []


from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam


bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

for l in bert_model.layers:
    l.trainable = True

x1_in = Input(shape=(None,))
x2_in = Input(shape=(None,))

x = bert_model([x1_in, x2_in])
x = Lambda(lambda x: x[:, 0])(x)
p = Dense(1, activation='sigmoid')(x)
model = Model([x1_in, x2_in], p)
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(1e-5), # 用足够小的学习率
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


model.fit_generator(
    train_D.__iter__(),
    steps_per_epoch=len(train_D),
    epochs=1,
    validation_data=valid_D.__iter__(),
    validation_steps=len(valid_D)
)

import shap

# we use the first 100 training examples as our background dataset to integrate over
backgroud_sample = data_generator(train_data, batch_size=100)
for i in backgroud_sample:
    backgroud_sample = i
    break

explainer = shap.DeepExplainer(model, backgroud_sample)

# explain the first 10 predictions
# explaining each prediction requires 2 * background dataset size runs
shap_values = explainer.shap_values(x_test[:10])

from keras_model import backend as K
seq_len=14

shape = y_pred[:, :, :].shape
out = K.get_value(K.ctc_decode(y_pred[:, :, :], input_length=np.ones(shape[0]) * shape[1])[0][0])[:,
      :seq_len]


from keras_bert import extract_embeddings

model_path = "./chinese_L-12_H-768_A-12/"
texts = ["今天天气不错啊！", "你明天想打篮球么？"]

embeddings = extract_embeddings(model_path, texts)

