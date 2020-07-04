# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from .pytorch_pretrained import BertModel, BertTokenizer


class Bert(nn.Module):
    def __init__(self, bert_pretrain_path, hidden_size, num_classes):
        super(Bert, self).__init__()
        self.bert = BertModel.from_pretrained(bert_pretrain_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, x_mask):
        """
        bert flow
        :param x: 输入的句子
        :param x_mask: 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        :return:
        """
        _, pooled = self.bert(x, attention_mask=x_mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out
