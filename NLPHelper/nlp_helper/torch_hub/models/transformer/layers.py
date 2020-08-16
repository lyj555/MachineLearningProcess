# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from .sub_layers import MultiHeadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, model_dim, n_head, hidden_size, dropout_rate, attention_dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(model_dim, n_head, dropout_rate,
                                                       attention_dropout)
        self.position_feed_forward = PositionwiseFeedForward(model_dim, hidden_size, dropout_rate)
        self.layer_norm = nn.LayerNorm(model_dim, eps=1e-6)

    def forward(self, seq_in, seq_mask=None):
        """
        encoder layer, two parts
        :param seq_in: [batch_size, seq_len, model_dim]
        :param seq_mask: None or [batch_size, seq_len, seq_len]
        :return: tuple([batch_size, seq_len, model_dim], [batch_size, n_head, seq_len, seq_len])
        """
        # [1]. first part
        seq_out, sim_score = self.multi_head_attention(seq_in, seq_in, seq_in, seq_mask)
        seq_in += seq_out  # residual connect
        seq_in = self.layer_norm(seq_in)  # layer normalization

        # [2]. second part
        pff = self.position_feed_forward(seq_in)
        seq_in += pff  # residual connect
        seq_in = self.layer_norm(seq_in)  # layer normalization
        return seq_in, sim_score


class DecodeLayer(nn.Module):
    def __init__(self, model_dim, n_head, hidden_size, dropout_rate, attention_dropout=0.1):
        super(DecodeLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(model_dim, n_head, dropout_rate, attention_dropout)
        self.enc_dec_multi_head_attention = MultiHeadAttention(model_dim, n_head, dropout_rate, attention_dropout)
        self.position_feed_forward = PositionwiseFeedForward(model_dim, hidden_size, dropout_rate)
        self.layer_norm = nn.LayerNorm(model_dim, eps=1e-6)

    def forward(self, enc_out, dec_in, enc_dec_mask=None, dec_mask=None):
        """
        decoder layer
        :param enc_out: [batch_size, seq_len1, model_dim]
        :param dec_in:  [batch_size, seq_len2, model_dim]
        :param enc_dec_mask: None or [batch_size, seq_len2, seq_len1]
        :param dec_mask: None or [batch_size, seq_len2, seq_len2]
        :return:
        """
        # [1]. first part, masked multi-head attention
        dec_out, dec_sim_score = self.multi_head_attention(dec_in, dec_in, dec_in, dec_mask)
        dec_in += dec_out  # residual connect
        dec_in = self.layer_norm(dec_in)  # layer normalization
        # dec_in'shape is [batch_size, seq_len1, model_dim]
        # dec_sim_score'shape is [batch_size, n_head, seq_len1, seq_len1]

        # [2] second part, encoder-decoder attention
        enc_dec_out, enc_dec_sim_score = \
            self.enc_dec_multi_head_attention(seq_in_query=dec_in, seq_in_key=enc_out,
                                              seq_in_value=enc_out, seq_mask=enc_dec_mask)
        dec_in += enc_dec_out  # residual connect
        dec_in = self.layer_norm(dec_in)  # layer normalization
        # dec_in'shape is [batch_size, seq

        # [3] third part, feed forward
        pff = self.position_feed_forward(dec_in)
        dec_in += pff  # residual connect
        dec_in = self.layer_norm(dec_in)  # layer normalization

        return dec_in, dec_sim_score, enc_dec_sim_score


if __name__ == "__main__":
    batch_size, n_head, seq_len, head_dim = 5, 2, 3, 4

    # 测试EncoderLayer
    ed = EncoderLayer(model_dim=8, n_head=2, hidden_size=20, dropout_rate=0.1)

    seq_in = torch.rand(batch_size, seq_len, n_head*head_dim)
    print(ed(seq_in)[0].shape)
    print(ed(seq_in)[1].shape)

    # 测试DecoderLayer
    dd = DecodeLayer(model_dim=8, n_head=2, hidden_size=20, dropout_rate=0.1)

    enc_out = torch.rand(batch_size, seq_len, n_head*head_dim)
    dec_in = torch.rand(batch_size, seq_len, n_head*head_dim)

    ret = dd(enc_out, dec_in)
    print(ret[0].shape)
    print(ret[1].shape)
    print(ret[2].shape)

    dec_in = torch.rand(batch_size, seq_len+3, n_head*head_dim)

    ret = dd(enc_out, dec_in)
    print(ret[0].shape)
    print(ret[1].shape)
    print(ret[2].shape)
