# -*- coding: utf-8 -*-

import torch


class DataIterator(object):
    def __init__(self, batch_data, batch_size, device):
        self.batch_size = batch_size
        self.batch_data = batch_data
        self.n_batches = len(batch_data) // batch_size

        self.residue = False  # 记录batch数量是否为整数
        if len(batch_data) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, data):
        """
        transform data to tensor
        :param data: list[tuple], tuple like (words_line, seq_len, y)
        :return:
        """
        x = torch.LongTensor([i[0] for i in data]).to(self.device)
        y = torch.LongTensor([i[2] for i in data]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        # seq_len = torch_model.LongTensor([_[1] for _ in data]).to(self.device)
        # return (x, seq_len), y
        return x, y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batch_data[self.index * self.batch_size: len(self.batch_data)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index > self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batch_data[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches
