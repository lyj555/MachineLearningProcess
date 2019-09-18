# -*- coding: utf-8 -*-

import torch.nn as nn


class TextRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout, num_classes, model_name="text_rnn"):
        super(TextRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=vocab_size-1)  # PAD's index
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.model_name = model_name

    def forward(self, x):
        x, _ = x
        out = self.embedding(x)  # [batch_size, seq_len, embedding_dim]=[128, 32, 300]
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out
