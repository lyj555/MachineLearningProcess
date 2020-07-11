# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LstmAttention(nn.Module):
    def __init__(self, hidden_size, num_layers, vocab_size, dropout, embedding_dim, padding_index,
                 embedding_pretrained=None):
        super(LstmAttention, self).__init__()

        self.embedding = \
            self._get_embedding(embedding_pretrained, vocab_size, embedding_dim, padding_index, **kwargs)

        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.tanh = nn.Tanh()
        self.K = nn.Parameter(torch.zeros(hidden_size * 2))
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        emb = self.embedding(x)  # [batch_size, seq_len, emb]
        V, _ = self.lstm(emb)  # [batch_size, seq_len, hidden_size*2]

        Q = self.tanh(V)  # rnn的tanh非线性激活, [batch_size, seq_len, hidden_size*2]
        # Q相当于是rnn的激活后的输出, K相当于是可学习的参数
        alpha = F.softmax(torch.matmul(Q, self.K), dim=1).unsqueeze(-1)  # 最后一个维度加1, [batch_size, seq_len, 1]
        out = V * alpha  # 每个位置乘以一个权重, [batch_size, seq_len, hidden_size*2]
        out = torch.sum(out, 1)  # 将所有句子的信息进行加和聚合, [batch_size, hidden_size*2]
        out = F.relu(out)  # [batch_size, hidden_size*2]
        out = self.fc1(out)  # [batch_size, hidden_size]
        out = self.fc(out)  # [batch_size, num_classes]
        return out

    @staticmethod
    def _get_embedding(embedding_pretrained, vocab_size, embedding_dim, padding_index, **kwargs):
        if embedding_pretrained is not None:
            assert isinstance(vocab_size, int) and isinstance(embedding_dim, int) and \
                   vocab_size >= 2 and embedding_dim >= 1, \
                "if embedding_pretrained is not specified, vocab_size and embedding_dim must specify."
        if embedding_pretrained is not None:
            embedding = nn.Embedding.from_pretrained(embedding_pretrained, **kwargs)
        else:
            embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim,
                                     padding_idx=padding_index, **kwargs)
        return embedding
