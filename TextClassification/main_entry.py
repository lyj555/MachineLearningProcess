# -*- coding: utf-8 -*-

import os
import pandas as pd
import torch
from sklearn.metrics import accuracy_score

from pre_process.format_content import format_content
from models.text_rnn import TextRNN
from train.torch_train import train, predict
from utils.split_data import split_data
from utils.torch_iterator import DataIterator

# data process param
RAW_DATA_PATH = "./data/THUCNews/train.txt"  # assume data format like `content_separator_label` and all samples in one file
BASE_MODEL_DIR = "./data/THUCNews/model"
if not os.path.exists(BASE_MODEL_DIR):
    os.makedirs(BASE_MODEL_DIR)
vocab_save_path = os.path.join(BASE_MODEL_DIR, "vocabulary.pkl")  # vocabulary save path
model_name = "text_rnn"
model_save_path = os.path.join(BASE_MODEL_DIR, f"{model_name}.ckpt")

batch_size = 128
min_word_freq = 2
pad_size = 32
max_vocab_size = 10000
line_sep = "\t"
random_state = 666

# text rnn param
embedding_dim = 300
hidden_size = 128
num_layers = 2
dropout = 0.5
num_classes = 10

# model train param
learning_rate = 1e-3
num_epochs = 10
early_stopping_batch = 100*10   # check loss every 100 batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# TODO: 1. use pre-trained word embedding 2. model initialize method 3. model save 4. add other model 5. other nlp task


def data_process(raw_data_path, line_sep):
    """
    do some data clean job, like word segmentation, etc
    :param raw_data_path: str, raw data path
    :param line_sep: str, line separator
    :return:
    """
    df = pd.read_csv(raw_data_path, header=None, names=["content", "label"], sep=line_sep)
    # omit the process of intermediate treatment
    raw_process_path = os.path.join(BASE_MODEL_DIR, "process_raw.txt")
    df.to_csv(raw_process_path, sep=line_sep, index=None, header=False, encoding="utf-8")
    return raw_process_path


def get_train_valid_test(data_path, line_sep, split_ratio, random_state):
    df = pd.read_csv(data_path, header=None, names=["content", "label"], sep=line_sep)
    train, valid, test = split_data(df, split_ratio, random_state)
    train_path = os.path.join(BASE_MODEL_DIR, "train.txt")
    train.to_csv(train_path, sep=line_sep, index=None, header=False, encoding="utf-8")
    valid_path = os.path.join(BASE_MODEL_DIR, "valid.txt")
    valid.to_csv(valid_path, sep=line_sep, index=None, header=False, encoding="utf-8")
    test_path = os.path.join(BASE_MODEL_DIR, "test.txt")
    test.to_csv(test_path, sep=line_sep, index=None, header=False, encoding="utf-8")
    return train_path, valid_path, test_path


if __name__ == "__main__":
    pre_path = data_process(RAW_DATA_PATH, line_sep)
    train_path, valid_path, test_path = get_train_valid_test(data_path=pre_path, line_sep=line_sep,
                                                             split_ratio=(0.6, 0.2, 0.2), random_state=random_state)

    train_pre_result, vocab_size = format_content(train_path, if_with_label=True, train_granularity="char", build_vocab=True,
                                                  vocab_save_path=vocab_save_path, max_vocab_size=max_vocab_size,
                                                  min_word_freq=min_word_freq, pad_size=pad_size, line_sep=line_sep)  # [(word_num_list, seq_len, y), (), ... ]

    valid_pre_result, _ = format_content(valid_path, if_with_label=True, train_granularity="char", build_vocab=False,
                                         vocab_save_path=vocab_save_path, max_vocab_size=max_vocab_size,
                                         min_word_freq=min_word_freq, pad_size=pad_size, line_sep=line_sep)

    test_pre_result, _ = format_content(test_path, if_with_label=True, train_granularity="char", build_vocab=False,
                                        vocab_save_path=vocab_save_path, max_vocab_size=max_vocab_size,
                                        min_word_freq=min_word_freq, pad_size=pad_size, line_sep=line_sep)

    train_iter = DataIterator(batch_data=train_pre_result, batch_size=batch_size, device=device)
    valid_iter = DataIterator(batch_data=valid_pre_result, batch_size=batch_size, device=device)
    test_iter = DataIterator(batch_data=test_pre_result, batch_size=batch_size, device=device)

    clf = TextRNN(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                  dropout=dropout, num_classes=num_classes)
    print(clf)
    optimizer = torch.optim.Adam(clf.parameters(), lr=learning_rate)
    train(train_iter=train_iter, dev_iter=valid_iter, model=clf, optimizer=optimizer, num_epochs=num_epochs,
          metric_func=accuracy_score, model_save_path=model_save_path, early_stopping_batch=early_stopping_batch)
    preds, labels = predict(model=clf, test_iter=test_iter, model_save_path=model_save_path)
    print("test metric", accuracy_score(y_pred=preds, y_true=labels))
