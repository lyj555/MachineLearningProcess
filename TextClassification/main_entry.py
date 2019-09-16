# -*- coding: utf-8 -*-

import os
import jieba
import pandas as pd

from pre_process.format_content import format_content

# assume data format like `content_separator_label` and all samples in one file
RAW_DATA_PATH = "./data/THUCNews/train.txt"


BASE_MODEL_DIR = "./data/THUCNews/model"
if not os.path.exists(BASE_MODEL_DIR):
    os.makedirs(BASE_MODEL_DIR)

# data process
df = pd.read_csv(RAW_DATA_PATH, header=None, names=["content", "label"], sep="\t")
# omit the process of intermediate treatment
raw_process_path = os.path.join(BASE_MODEL_DIR, "process_raw.txt")
df.to_csv(raw_process_path, sep="\t", index=None, header=False, encoding="utf-8")

vocab_save_path = os.path.join(BASE_MODEL_DIR, "vocabulary.pkl")
pre_result = format_content(raw_process_path, if_with_label=True, train_granularity="char", build_vocab=True,
                            vocab_save_path=vocab_save_path, max_vocab_size=100,
                            min_word_freq=2, pad_size=32, line_sep="\t")
train_pre, valid_pre, test_pre = split_dataset(pre_result)

train_iter, valid_iter, test_iter = build_iterator(train_pre, valid_pre, test_pre, batch_size=4)

clf = train_by_torch(train_iter, valid_iter, model, optimizer, model_save_path)

pred, label = predict(clf, test_iter)

# pre_un = pre_process(unseen_sample)
# un_iter = build_iterator(pre_un)
# pred, _ = predict(clf, test_iter)
