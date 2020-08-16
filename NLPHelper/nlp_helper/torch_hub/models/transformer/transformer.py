# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn

from .layers import EncoderLayer, DecodeLayer


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, n_position):
        super(PositionalEncoding, self).__init__()
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / model_dim)) for i in range(model_dim)]
                                for pos in range(n_position)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])

    def forward(self, x):
        x = x + nn.Parameter(self.pe, requires_grad=False)
        return x


class Encoder(nn.Module):
    def __init__(self, seq_len, n_layers, n_head, hidden_size, dropout_rate, n_vocab, model_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(n_vocab, model_dim)
        self.position_embedding = PositionalEncoding(model_dim, seq_len)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(model_dim, n_head, hidden_size, dropout_rate)
            for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(model_dim, eps=1e-6)

    def forward(self, src_seq, src_mask, return_attns=False):
        """
        encode block, including many encoding layers
        :param src_seq: [batch_size, seq_len], source input
        :param src_mask: [batch_size, seq_len, seq_len]
        :param return_attns:
        :return:
        """
        src_emb = self.embedding(src_seq)
        src_emb = self.position_embedding(src_emb)
        src_emb = self.dropout(src_emb)
        src_emb = self.layer_norm(src_emb)

        enc_attns = []
        for enc_layer in self.layer_stack:
            src_emb, src_attn = enc_layer(src_emb, src_mask)
            if return_attns:
                enc_attns.append(src_attn)

        if return_attns:
            return src_emb, enc_attns
        else:
            return src_emb,


class Decoder(nn.Module):
    def __init__(self, seq_len, n_layers, n_head, hidden_size, dropout_rate, n_vocab, model_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(n_vocab, model_dim)
        self.position_embedding = PositionalEncoding(model_dim, seq_len)
        self.layer_stack = nn.ModuleList([
            DecodeLayer(model_dim, n_head, hidden_size, dropout_rate)
            for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(model_dim, eps=1e-6)

    def forward(self, trg_seq, enc_out, trg_mask, enc_dec_mask, return_attns=False):
        trg_emb = self.embedding(trg_seq)
        trg_emb = self.position_embedding(trg_emb)
        trg_emb = self.dropout(trg_emb)
        trg_emb = self.layer_norm(trg_emb)

        dec_attns, enc_dec_attns = [], []
        for dec_layer in self.layer_stack:
            trg_emb, dec_attn, enc_dec_attn = \
                dec_layer(enc_out, trg_emb, enc_dec_mask, trg_mask)
            if return_attns:
                dec_attns.append(dec_attn)
                enc_dec_attns.append(enc_dec_attn)

        if return_attns:
            return trg_emb, dec_attns, enc_dec_attns
        else:
            return trg_emb,


def get_pad_mask(seq, pad_idx=0):
    """
    获取pad mask, 表示句子中需要mask掉的部分
    :param seq: [batch_size, seq_len]
    :param pad_idx: int
    :return: [batch_size, 1, seq_len]
    """
    return (seq != pad_idx).unsqueeze(-2)


def get_sequence_mask(seq):
    """
    sequence mask, 每个位置只可以看到其前面的信息
    :param seq: [batch_size, seq_len]
    :return: [1, seq_len, seq_len], 下对角矩阵
    """
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class Transformer(nn.Module):
    def __init__(self, src_len, trg_len, n_src_vocab, n_trg_vocab, model_dim, n_layers, n_head, hidden_size, dropout_rate=0.1, pad_idx=0):
        super(Transformer, self).__init__()
        self.pad_idx = pad_idx

        self.encoder = Encoder(seq_len=src_len, n_layers=n_layers, n_head=n_head, hidden_size=hidden_size,
                               dropout_rate=dropout_rate, n_vocab=n_src_vocab, model_dim=model_dim)
        self.decoder = Decoder(seq_len=trg_len, n_layers=n_layers, n_head=n_head, hidden_size=hidden_size,
                               dropout_rate=dropout_rate, n_vocab=n_trg_vocab, model_dim=model_dim)
        self.trg_word_prj = nn.Linear(model_dim, n_trg_vocab, bias=False)

    def forward(self, src_seq, trg_seq):
        src_mask = get_pad_mask(src_seq, self.pad_idx)
        trg_mask = get_sequence_mask(trg_seq) & get_pad_mask(trg_seq, self.pad_idx)

        enc_out, *_ = self.encoder(src_seq, src_mask)
        dec_out, *_ = self.decoder(trg_seq, enc_out, trg_mask, src_mask)

        seq_out = self.trg_word_prj(dec_out)  # [batch_size, seq_len, n_trg_vocab]

        return seq_out


if __name__ == "__main__":
    # 测试Encoder部分
    en = Encoder(seq_len=10, n_layers=4, n_head=3, hidden_size=30, dropout_rate=0.1,
                 n_vocab=100, model_dim=12)
    src_seq = torch.randint(low=0, high=50, size=(5, 10))
    ret = en(src_seq, src_mask=None, return_attns=True)
    print(ret[0].shape)
    print(len(ret[1]))
    print(ret[1][0].shape)

    # 测试mask
    en = Encoder(seq_len=10, n_layers=4, n_head=3, hidden_size=30, dropout_rate=0.1,
                 n_vocab=100, model_dim=12)
    src_seq = torch.randint(low=0, high=50, size=(5, 10))
    src_mask = torch.randint(low=0, high=2, size=(5, 10, 10))
    ret = en(src_seq, src_mask=src_mask, return_attns=True)
    print(ret[0].shape)
    print(len(ret[1]))
    print(ret[1][0].shape)

    # 测试Decoder部分
    dn = Decoder(seq_len=10, n_layers=4, n_head=3, hidden_size=30, dropout_rate=0.1,
                 n_vocab=100, model_dim=12)
    trg_seq = torch.randint(low=0, high=50, size=(5, 10))
    enc_out = torch.rand(5, 9, 12)

    ret = dn(trg_seq, enc_out, None, None, True)
    print(ret[0].shape)
    print(len(ret[1]))
    print(ret[1][0].shape)
    print(ret[2][0].shape)

    # mask
    trg_mask = torch.randint(low=0, high=2, size=(5, 10, 10))
    enc_dec_mask = torch.randint(low=0, high=2, size=(5, 10, 9))

    ret = dn(trg_seq, enc_out, trg_mask, enc_dec_mask, True)
    print(ret[0].shape)
    print(len(ret[1]))
    print(ret[1][0].shape)
    print(ret[2][0].shape)

    # 测试transformer
    tf = Transformer(src_len=10, trg_len=5, n_src_vocab=200, n_trg_vocab=100,
                     model_dim=300, n_layers=6, n_head=10, hidden_size=128)

    src_seq = torch.randint(low=0, high=200, size=(64, 10))
    trg_seq = torch.randint(low=0, high=100, size=(64, 5))

    tf_out = tf(src_seq, trg_seq)
    print(tf_out.shape)  # 64, 5, 100
