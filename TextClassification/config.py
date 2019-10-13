# -*- coding: utf-8 -*-

import os
from datetime import datetime
from sklearn.metrics import accuracy_score

line_sep = "\t"
random_state = 666

BASE_MODEL_DIR = "./data/THUCNews/model"
LOG_PATH = "./data/THUCNews/model/log.txt"
SAVE_RESULT = True
if not os.path.exists(BASE_MODEL_DIR):
    os.makedirs(BASE_MODEL_DIR)
vocab_save_path = os.path.join(BASE_MODEL_DIR, "vocabulary.pkl")  # vocabulary save path

train_granularity = "char"
model_name = "text_rnn"
model_save_path = os.path.join(BASE_MODEL_DIR, f"{model_name}_{train_granularity}_"
                                               f"{datetime.now().strftime('%Y%m%d_%H%M')}.ckpt")


PARAM = {
    "framework": "torch",
    "data_process": {"raw_data_path": "./data/THUCNews/model_component.txt",
                     "line_sep": line_sep},
    "get_train_valid_test": {"split_ratio": (0.6, 0.2, 0.2), "random_state": random_state, "line_sep": line_sep},
    "format_content": {"train_granularity": train_granularity, "vocab_save_path": vocab_save_path,
                       "max_vocab_size": 10000, "min_word_freq": 2, "pad_size": 32,
                       "line_sep": line_sep},
    "DataIterator": {"batch_size": 128},
    "model": {"model_name": "text_rnn", "embedding_dim": 300, "hidden_size": 128, "num_layers": 2,
              "dropout": 0.5, "num_classes": 10},
    "model_component": {"learning_rate": 1e-3, "num_epochs": 1, "early_stopping_batch": 100*10, "metric_func": accuracy_score,
              "model_save_path": model_save_path}
}

