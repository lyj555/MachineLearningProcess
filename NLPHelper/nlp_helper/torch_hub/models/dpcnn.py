# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DPCNN(nn.Module):
    def __init__(self, num_filters, num_classes, vocab_size, embedding_dim, padding_index=0,
                 embedding_pretrained=None, **kwargs):
        super(DPCNN, self).__init__()

        self.embedding = \
            self._get_embedding(embedding_pretrained, vocab_size, embedding_dim, padding_index, **kwargs)

        self.conv_region = nn.Conv2d(1, num_filters, (3, embedding_dim), stride=1)
        self.conv = nn.Conv2d(num_filters, num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # [batch_size, seq_len, emb]
        x = x.unsqueeze(1)  # [batch_size, 1, seq_len, emb]  conv2D输入的input channel是1,所以加入一个维度
        # [1] 先做一次region embedding, 相当于是textcnn的部分
        x = self.conv_region(x)  # [batch_size, num_filters, seq_len-3+1, 1]

        # [2] 然后做两次等长卷积, 相当于做信息合并
        x = self.padding1(x)  # top和bottom补[PAD], [batch_size, num_filters, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, num_filters, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, num_filters, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, num_filters, seq_len-3+1, 1]

        # [3] 金字塔卷积
        while x.size()[2] > 2:
            x = self._block(x)
        x = x.squeeze()  # 去掉等于1的维度, [batch_size, num_filters]
        x = self.fc(x)  # [batch_size, num_classes]
        return x

    def _block(self, x):
        # [1] 做了一次下采样
        x = self.padding2(x)  # bottom补[PAD], [batch_size, num_filters, n, 1] --> [batch_size, num_filters, n+1, 1]
        px = self.max_pool(x)  # 1/2下采样, [batch_size, num_filters, n, 1] --> [batch_size, num_filters, n//2-1, 1]

        # [2] 做了两次等长卷积
        x = self.padding1(px)  # [batch_size, num_filters, n, 1] --> [batch_size, num_filters, n+2, 1]
        x = F.relu(x)
        x = self.conv(x)  # [batch_size, num_filters, n, 1] --> [batch_size, num_filters, n-3+1, 1]
        x = self.padding1(x)  # [batch_size, num_filters, n, 1] --> [batch_size, num_filters, n+2, 1]
        x = F.relu(x)
        x = self.conv(x)  # [batch_size, num_filters, n, 1] --> [batch_size, num_filters, n-3+1, 1]

        # [3] 残差连接
        x = x + px
        return x

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


if __name__ == "__main__":
    dpcnn = DPCNN(num_filters=64, num_classes=2, vocab_size=100, embedding_dim=30)
    aa = torch.LongTensor(np.random.randint(5, size=(2, 9)))
    dpcnn(aa)
