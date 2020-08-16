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
        scaled dot-product attention, query, key&value
        softmax(q*k^T/scale)*v,
        q*k^T, 可以表示为seq_len1句子维度下, seq_len1的每一个位置用seq_len2维度来表示,
        但是并不是每个seq_len2的维度都可以利用, 如果用pad, 那么需要将seq_len2中pad部分mask掉,
        如果有sequence_mask, 那么需要将seq_len2后面的部分mask掉.
        mask的部分用一个大的数字代替, 经过softmax后变为0, 然后乘以value的时候, 间接达到了mask的作用
        :param q: [batch_size, n_head, seq_len1, head_dim]
        :param k: [batch_size, n_head, seq_len2, head_dim]
        :param v: [batch_size, n_head, seq_len2, head_dim]
        :param mask: None or [batch_size, seq_len1, seq_len2]
        :return:
        """
        # [1]. 计算Q*K^T
        sim_score = torch.matmul(q, k.transpose(2, 3))  # [batch_size, n_head, seq_len1, seq_len2]
        # 相当于query中的每个字和其余位置做点积, 所以最后的维度为seq_len1*seq_len2
        # [2]. scale放缩
        sim_score = sim_score / self.scale  # [batch_size, n_head, seq_len1, seq_len2]

        # [3]. 引入mask, 变换为一个大的负数
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len1, seq_len2]
            sim_score = sim_score.masked_fill(mask == 0, -1e9)

        # [3]. softmax, 压缩为概率分布
        sim_score = F.softmax(sim_score, dim=-1)  # [batch_size, n_head, seq_len1, seq_len2]

        # [4]. 添加dropout
        sim_score = self.dropout(sim_score)  # [batch_size, n_head, seq_len1, seq_len2]

        # [5]. 获取attention value
        self_attn = torch.matmul(sim_score, v)  # [batch_size, n_head, seq_len1, head_dim]
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

    def forward(self, seq_in_query, seq_in_key, seq_in_value, seq_mask=None):
        """
        sequence input, 注意seq_in_query和seq_in_key的shap可能不能,
        做self-attention时, 此时三者的输入的维度是一致的;
        做encode-decode attention时, `seq_in_query`维度和另外两个可能不一致
        :param seq_in_query: [batch_size, seq_len1, model_dim]
        :param seq_in_key: [batch_size, seq_len2, model_dim]
        :param seq_in_value: [batch_size, seq_len2, model_dim]
        :param seq_mask: None or [batch_size, seq_len1, seq_len2]
        :return: tuple([batch_size, seq_len1, model_dim], [batch_size, n_head, seq_len1, seq_len2])
        """
        batch_size = seq_in_query.size()[0]

        # [1]. 首先将input sequence变化为key, query&value, 然后变换为多头的形态
        query = self.fc_q(seq_in_query)\
            .view(batch_size, self.n_head, -1, self.head_dim)  # [batch_size, n_head, seq_len1, head_dim]
        key = self.fc_k(seq_in_key)\
            .view(batch_size, self.n_head, -1, self.head_dim)  # [batch_size, n_head, seq_len2, head_dim]
        value = self.fc_v(seq_in_value)\
            .view(batch_size, self.n_head, -1, self.head_dim)  # [batch_size, n_head, seq_len2, head_dim]

        # [2]. self-attention部分
        self_attn, sim_score = self.scaled_dot_product(query, key, value, mask=seq_mask)
        # self_attn: [batch_size, n_head, seq_len1, head_dim],  sim_score: [batch_size, n_head, seq_len1, seq_len2]

        # [3]. 由多头的形态还原为sequence的形状
        self_attn = self_attn.transpose(1, 2)\
            .reshape(batch_size, -1, self.n_head*self.head_dim)  # [batch_size, seq_len1, n_head*n_dim]
        self_attn = self.dropout(self_attn)
        return self_attn, sim_score


class PositionwiseFeedForward(nn.Module):
    def __init__(self, model_dim, hidden_size, dropout_rate=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(model_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, model_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        position-wise feed forward
        :param x: [batch_size, seq_len, model_dim]
        :return:
        """
        x = self.fc1(x)  # [batch_size, seq_len, hidden_size]
        x = F.relu(x)
        x = self.fc2(x)  # [batch_size, seq_len, model_dim]
        x = self.dropout(x)
        return x


if __name__ == "__main__":
    # 测试ScaledDotProductAttention
    batch_size, n_head, seq_len, head_dim = 5, 2, 3, 4
    sd = ScaledDotProductAttention(0.5, 0.1)

    q = torch.rand(batch_size, n_head, seq_len, head_dim)
    k = torch.rand(batch_size, n_head, seq_len, head_dim)
    v = torch.rand(batch_size, n_head, seq_len, head_dim)

    ret = sd(q, k, v)
    print(ret[0].shape)
    print(ret[1].shape)

    # 测试MultiHeadAttention
    seq_in = torch.rand(batch_size, seq_len, n_head*head_dim)
    seq_mask = torch.randint(low=0, high=2, size=(batch_size, seq_len, seq_len))

    # normal input
    md = MultiHeadAttention(8, 2, 0.5)
    ret = md(seq_in, seq_in, seq_in)
    print(ret[0].shape)
    print(ret[1].shape)

    # normal input, seq_mask
    ret = md(seq_in, seq_in, seq_in, seq_mask)
    print(ret[0].shape)
    print(ret[1].shape)

    # different input
    md = MultiHeadAttention(8, 2, 0.5)
    seq_in1 = torch.rand(batch_size, seq_len, n_head*head_dim)
    seq_in2 = torch.rand(batch_size, seq_len+3, n_head*head_dim)
    ret = md(seq_in2, seq_in1, seq_in1)
    print(ret[0].shape)
    print(ret[1].shape)

    # different input, seq_mask
    md = MultiHeadAttention(8, 2, 0.5)
    seq_in1 = torch.rand(batch_size, seq_len, n_head*head_dim)
    seq_in2 = torch.rand(batch_size, seq_len+3, n_head*head_dim)
    seq_mask = torch.randint(low=0, high=2, size=(batch_size, seq_len+3, seq_len))
    ret = md(seq_in2, seq_in1, seq_in1, seq_mask)
    print(ret[0].shape)
    print(ret[1].shape)

    # 测试PositionwiseFeedForward
    seq_in = torch.rand(batch_size, seq_len, n_head * head_dim)
    pf = PositionwiseFeedForward(model_dim=n_head*head_dim, hidden_size=30)

    print(pf(seq_in).shape)
