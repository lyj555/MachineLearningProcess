# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextRCNN(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout, num_classes, vocab_size=None, embedding_dim=None,
                 embedding_pretrained=None, padding_index=0, **kwargs):
        super(TextRCNN, self).__init__()

        self.embedding = \
            self._get_embedding(embedding_pretrained, vocab_size, embedding_dim, padding_index, **kwargs)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers,
                            bidirectional=True, batch_first=True)
        # self.maxpool = nn.MaxPool1d(seq_len)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2 + embedding_dim, num_classes)

    def forward(self, x):
        embed = self.embedding(x)  # [batch_size, seq_len, emb]
        out, _ = self.lstm(embed)  # [batch_size, seq_len, 2*hidden_size] 2表示双向
        out = torch.cat((embed, out), 2)  # [batch_size, seq_len, emb+2*hidden_size]
        out = F.relu(out)
        out = out.permute(0, 2, 1)  # [batch_size, emb+2*hidden_size, seq_len]
        # out = self.maxpool(out).squeeze()
        out = F.max_pool1d(out, out.size(2)).squeeze(2)  # [batch_size, emb+2*hidden_size]
        out = self.dropout(out)
        out = self.fc(out)  # [batch_size, 2]
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


# import numpy as np
# tt = TextRCNN(10, 2, 0.5, 2, 50, 30)
# aa = torch.LongTensor(np.random.randint(5, size=(2, 3)))
# tt(aa)
