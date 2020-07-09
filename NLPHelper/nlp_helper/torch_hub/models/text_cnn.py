# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, num_filters, filter_sizes, dropout, num_classes, vocab_size=None, embedding_dim=None,
                 embedding_pretrained=None, padding_index=0, **kwargs):
        super(TextCNN, self).__init__()

        self.embedding = \
            self._get_embedding(embedding_pretrained, vocab_size, embedding_dim, padding_index, **kwargs)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, embedding_dim)) for k in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        out = self.embedding(x)  # [batch_size, seq_len, emb_size]
        out = out.unsqueeze(1)  # 增加一个channel, 对应nn.Conv2D的1, [batch_size, 1, seq_len, emb_size]
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)  # 按照axis=1合并, [batch_size, len(filter_size)*num_filters]
        out = self.dropout(out)  # 保持相同维度[batch_size, len(filter_size)*num_filters]
        out = self.fc(out)  # [batch_size, num_classes]
        return out

    @staticmethod
    def conv_and_pool(x, conv):
        x = F.relu(conv(x)).squeeze(3)  # 第三个维度是1, 将其压缩, [batch_size, num_filters, seq_len-k+1(滑动窗口)]
        x = F.max_pool1d(x, x.size(2)).squeeze(2)  # 沿着最后一个维度取最值,  最后一个维度取不同的filter_size时维度不同,
        # [batch_size, num_filters, 1]->[batch_size, num_filters]
        return x

    @staticmethod
    def _check_cnn_param(num_filters, filter_sizes):
        assert isinstance(num_filters, int) and num_filters >= 1, \
            "num_filters must be integer number and greater than 1"
        assert isinstance(filter_sizes, (list, tuple)), "filter_sizes must be list or tuple"

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
