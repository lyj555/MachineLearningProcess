# -*- coding: utf-8 -*-

import os
import numpy as np
from datetime import datetime
import pandas as pd
from keras import layers
from keras.models import Sequential
from keras import optimizers

from pre_process.format_content import format_content
from utils.split_data import split_data
from config import PARAM, BASE_MODEL_DIR, SAVE_RESULT, LOG_PATH


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
    train_path = os.path.join(BASE_MODEL_DIR, "model_component.txt")
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

    n_classes = 10
    train_x = np.array([np.array(i[0]) for i in train_pre_result], dtype=int)
    train_y = np.array([np.array(i[2]) for i in train_pre_result], dtype=int)
    train_y = np.eye(n_classes, dtype=int)[train_y]

    valid_x = np.array([np.array(i[0]) for i in valid_pre_result], dtype=int)
    valid_y = np.array([np.array(i[2]) for i in valid_pre_result], dtype=int)
    valid_y = np.eye(n_classes, dtype=int)[valid_y]

    test_x = np.array([np.array(i[0]) for i in test_pre_result], dtype=int)
    test_y = np.array([np.array(i[2]) for i in test_pre_result], dtype=int)
    test_y = np.eye(n_classes, dtype=int)[test_y]

    # model = Sequential()
    # model.add(layers.Embedding(input_dim=vocab_size, output_dim=300, input_length=32))
    # model.add(layers.LSTM(32))
    # model.add(layers.Dense(10, activation="softmax"))

    model = Sequential()
    model.add(layers.Embedding(input_dim=vocab_size, output_dim=300, input_length=32))
    model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True)))
    model.add(layers.Bidirectional(layers.LSTM(32)))
    model.add(layers.Dense(10, activation="softmax"))

    model.summary()

    optimizer = optimizers.Adam(lr=1e-3)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(x=train_x[:10000], y=train_y[:10000], batch_size=128, epochs=1, verbose=1,
                        validation_data=(valid_x, valid_y))

    import shap

    # we use the first 100 training examples as our background dataset to integrate over
    explainer = shap.DeepExplainer(model, x_train[:100])

    # explain the first 10 predictions
    # explaining each prediction requires 2 * background dataset size runs
    shap_values = explainer.shap_values(x_test[:10])
    shap.image_plot()

    # if SAVE_RESULT:
    #     df_ret = pd.DataFrame({"model_name": [PARAM["model"]["model_name"]],
    #                            "train_time": [cost_sec],
    #                            "test_metric": [test_metric],
    #                            "train_ret": [train_ret],
    #                            "valid_ret": [valid_ret],
    #                            "param": [PARAM],
    #                            "framework": [PARAM["framework"]]})
    #     df_ret = df_ret[["model_name", "framework", "train_time", "test_metric", "param", "train_ret", "valid_ret"]]
    #     if not os.path.exists(LOG_PATH):
    #         df_ret.to_csv(LOG_PATH, sep="\t", index=None)
    #     else:
    #         df_ret.to_csv(LOG_PATH, sep="\t", index=None, header=False, mode="a+")

