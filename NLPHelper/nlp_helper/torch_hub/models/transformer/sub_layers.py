# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale, attention_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, q, k, v, mask=None):
        """
        scaled dot-product attention,
        softmax(q*k^T/scale)*v
        :param q: [batch_size, n_head, seq_len, head_dim]
        :param k: [batch_size, n_head, seq_len, head_dim]
        :param v: [batch_size, n_head, seq_len, head_dim]
        :param mask:
        :return:
        """
        # [1]. 计算Q*K^T
        sim_score = torch.matmul(q, k.transpose(2, 3))  # [batch_size, n_head, seq_len, seq_len]

        # [2]. scale放缩
        sim_score = sim_score / self.scale  # [batch_size, n_head, seq_len, seq_len]

        # [3]. 引入mask, 变换为一个大的负数
        if mask is not None:
            sim_score = sim_score.masked_fill(mask == 0, -1e9)

        # [3]. softmax, 压缩为概率分布
        sim_score = F.softmax(sim_score, dim=-1)  # [batch_size, n_head, seq_len, seq_len]

        # [4]. 添加dropout
        sim_score = self.dropout(sim_score)  # [batch_size, n_head, seq_len, seq_len]

        # [5]. 获取attention value
        self_attn = torch.matmul(sim_score, v)  # [batch_size, n_head, seq_len, head_dim]
        return self_attn, sim_score


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, n_head, dropout_rate, attention_dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.model_dim = model_dim
        self.n_head = n_head
        self.head_dim = self.model_dim // self.n_head  # 将model_dim分为n_head份, 查看每份的长度

        self.fc_q = nn.Linear(self.model_dim, self.head_dim * self.n_head, bias=False)
        self.fc_k = nn.Linear(self.model_dim, self.head_dim * self.n_head, bias=False)
        self.fc_v = nn.Linear(self.model_dim, self.head_dim * self.n_head, bias=False)

        self.scaled_dot_product = ScaledDotProductAttention(scale=model_dim**0.5, attention_dropout=attention_dropout)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, seq_in, seq_mask=None):
        batch_size = seq_in.size()[0]

        key = self.fc_k(seq_in).view(batch_size, self.n_head, -1, self.head_dim)  # [batch_size, n_head, seq_len, head_dim]
        query = self.fc_q(seq_in).view(batch_size, self.n_head, -1, self.head_dim)  # [batch_size, n_head, seq_len, head_dim]
        value = self.fc_v(seq_in).view(batch_size, self.n_head, -1, self.head_dim)  # [batch_size, n_head, seq_len, head_dim]

        self_attn, sim_score = self.scaled_dot_product(query, key, value, mask=seq_mask)
        # self_attn: [batch_size, n_head, seq_len, head_dim],  sim_score: [batch_size, n_head, seq_len, seq_len]

        self_attn = self_attn.transpose(1, 2).reshape(batch_size, -1, self.n_head*self.head_dim)  # [batch_size, seq_len, n_head*n_dim]
        self_attn = self.dropout(self_attn)
        return self_attn, sim_score


class PositionwiseFeedForward(nn.Module):
    def __init__(self, model_dim, hidden_size):
        super(PositionwiseFeedForward, self).__init__()



if __name__ == "__main__":
    batch_size, n_head, seq_len, head_dim = 5, 2, 3, 4
    sd = ScaledDotProductAttention(0.5, 0.1)

    q = torch.rand(batch_size, n_head, seq_len, head_dim)
    k = torch.rand(batch_size, n_head, seq_len, head_dim)
    v = torch.rand(batch_size, n_head, seq_len, head_dim)

    ret = sd(q, k, v)
    print(ret[0].shape)
    print(ret[1].shape)

    seq_in = torch.rand(batch_size, seq_len, n_head*head_dim)

    md = MultiHeadAttention(8, 2, 0.5)
    ret = md(seq_in)
    print(ret[0].shape)
    print(ret[1].shape)
