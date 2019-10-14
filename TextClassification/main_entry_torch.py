# -*- coding: utf-8 -*-

import os
from datetime import datetime
import pandas as pd
import torch

from pre_process.format_content import format_content
from torch_model.models.text_rnn import TextRNN
from torch_model.torch_train import train, predict
from utils.split_data import split_data
from torch_model.torch_iterator import DataIterator
from config import PARAM, BASE_MODEL_DIR, SAVE_RESULT, LOG_PATH


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# TODO: 1. use pre-trained word embedding 2. model initialize method
# TODO 3. model save 4. add other model 5. other nlp task 6. batch from file


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
    # [1] data process, like word segmentation, transform data format, etc.
    pre_path = data_process(**PARAM["data_process"])

    # [2] split data into model_component/valid/test, can be cut into multiple data sets as needed
    train_path, valid_path, test_path = get_train_valid_test(data_path=pre_path, **PARAM["get_train_valid_test"])

    # [3] according to cutting data, format the content, including cut or pad the content by pad_size or build vocabulary etc.
    train_pre_result, vocab_size = format_content(train_path, build_vocab=True, **PARAM["format_content"])  # [(word_num_list, seq_len, y), (), ... ]

    valid_pre_result, _ = format_content(valid_path, build_vocab=False, **PARAM["format_content"])

    test_pre_result, _ = format_content(test_path, build_vocab=False, **PARAM["format_content"])

    # [4] create data iterator by cutting data
    batch_size = PARAM["DataIterator"]["batch_size"]
    train_iter = DataIterator(batch_data=train_pre_result, batch_size=batch_size, device=device)
    valid_iter = DataIterator(batch_data=valid_pre_result, batch_size=batch_size, device=device)
    test_iter = DataIterator(batch_data=test_pre_result, batch_size=batch_size, device=device)

    # [5] initialize your model
    clf = TextRNN(vocab_size=vocab_size, **PARAM["model"])
    print(clf)

    # [6] start training
    start_time = datetime.now()
    train_ret, valid_ret = train(train_iter=train_iter, dev_iter=valid_iter, model=clf, **PARAM["model_component"])
    end_time = datetime.now()
    cost_sec = (end_time-start_time).seconds
    print(f"all spend {cost_sec} seconds.")

    # [7] check test set metric
    preds, labels = predict(model=clf, test_iter=test_iter, model_save_path=PARAM["model_component"]["model_save_path"])
    test_metric = PARAM["model_component"]["metric_func"](y_pred=preds, y_true=labels)
    print("test metric", test_metric)

    if SAVE_RESULT:
        df_ret = pd.DataFrame({"model_name": [PARAM["model"]["model_name"]],
                               "train_time": [cost_sec],
                               "test_metric": [test_metric],
                               "train_ret": [train_ret],
                               "valid_ret": [valid_ret],
                               "param": [PARAM],
                               "framework": [PARAM["framework"]]})
        df_ret = df_ret[["model_name", "framework", "train_time", "test_metric", "param", "train_ret", "valid_ret"]]
        if not os.path.exists(LOG_PATH):
            df_ret.to_csv(LOG_PATH, sep="\t", index=None)
        else:
            df_ret.to_csv(LOG_PATH, sep="\t", index=None, header=False, mode="a+")


    # load model
    # clf = TextRNN(vocab_size=vocab_size, **PARAM["model"])
    # clf.load_state_dict(torch_model.load(PARAM["model_component"]["model_save_path"]))
    #
    # import shap
    # import numpy as np
    #
    # for x, y in train_iter:
    #     train_x = x[:100]
    #     train_y = y[:100]
    #     break
    #
    # for x, y in test_iter:
    #     test_x = x[:100]
    #     test_y = y[:100]
    #     break
    #
    # explainer = shap.DeepExplainer(clf, train_x)
    # shap_values = explainer.shap_values(test_x)
