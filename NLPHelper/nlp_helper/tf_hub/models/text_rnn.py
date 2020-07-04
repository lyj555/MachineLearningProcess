# -*- coding: utf-8 -*-

from keras import Input, Model
from keras.layers import Embedding, Bidirectional, LSTM, GRU, Dense


def text_rnn(vocab_size, max_len, embedding_dim, class_num,
             rnn_block="lstm", bidirection=True, layer_num=2, **kwargs):
    assert rnn_block in ("lstm", "gru"), "rnn_block must be in ('lstm', 'gru', )"
    assert isinstance(class_num, int), "class_num must be an Integer number!"
    last_activation = "sigmoid" if class_num == 2 else "softmax"

    inputs = Input((max_len, ))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                  input_length=max_len)(inputs)
    for i in range(layer_num):
        if i != layer_num - 1:
            rnn_block = LSTM(return_sequences=True, **kwargs) if rnn_block == "lstm" else \
                GRU(return_sequences=True, **kwargs)
        else:
            rnn_block = LSTM(return_sequences=False, **kwargs) if rnn_block == "lstm" else \
                GRU(return_sequences=False, **kwargs)
        if bidirection:
            x = Bidirectional(rnn_block)(x)
        else:
            x = rnn_block(x)
    output = Dense(class_num, activation=last_activation)(x)
    model = Model(inputs=inputs, output=output)
    return model
