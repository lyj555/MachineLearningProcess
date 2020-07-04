# -*- coding: utf-8 -*-

import numpy as np


class SelfIterator(object):
    def __init__(self, batch_data, batch_size, shuffle=False, sampler=None, drop_last=False,
                 device=None, to_tensor=False, tensor_type=None):
        """
        根据输入的数据生成按照batch_size迭代器
        :param batch_data: tuple, 输入的数据，可以numpy格式，可以是tensor格式，如果一份数据，就是(x, )
        :param batch_size: int, 每次迭代取的数据样本大小
        :param shuffle: bool, 每次初始化迭代器时是否打乱数据，默认为False
        :param sampler: None or callable, 指定每次初始化迭代器时的采样策略
        :param drop_last: bool, 如果迭代最后剩余的数据不足一个batch，指定是否去掉，默认为True
        :param device: 如果转换数据为torch类型，指定GPU和CPU
        :param to_tensor: bool, 是否生成数据时转换为tensor类型
        :param tensor_type: tuple[Tensor.Type], 如果to_tensor为True，必须指定输入的每一部分batch_data的类型
        """
        self.batch_data, self.batch_part, self.data_size = self._check_batch_data(batch_data)
        self.batch_size = batch_size
        # self.batch_data = batch_data
        self.n_batches = self.data_size // self.batch_size

        if self.n_batches > 0:
            self._residue = False if self.data_size % self.n_batches == 0 else True  # 记录batch数量是否为整数
        else:  # 不足一个batch
            self._residue = True
        self.index = 0
        self.shuffle = shuffle  # 是否打乱样本
        self.sampler = sampler  # 按照指定的策略进行样本的选取
        self.drop_last = drop_last  # 是否保留最后的不足一个batch的数据
        self.to_tensor = to_tensor  # 是否将数据转换为tensor结构
        self.tensor_type = tensor_type  # 指定每个数据的tensor类型
        self.device = device  # 数据类型，CPU or GPU
        self._check_to_tensor()  # 校验输入的tensor_type是否合法
        self._take_sample()

    def _take_sample(self):
        assert self.sampler is None or callable(self.sampler), "sampler must be None or callable object!"
        assert isinstance(self.shuffle, bool), "param shuffle must be a bool object"
        if self.sampler is None and self.shuffle:
            self.sampler = self.shuffle_sampler

        if self.sampler is not None:
            indexes = self.sampler(self.batch_data)
            assert len(indexes) == self.data_size, \
                f"the sampler returned size is {len(indexes)}, but input data_size is {self.data_size}."
            try:
                self.batch_data = [np.array(i)[indexes] for i in self.batch_data]
            except IndexError:
                raise ValueError("the sampler returned invalid indexes,")

    @staticmethod
    def shuffle_sampler(batch_data):
        data_size = len(batch_data[0])
        indexes = np.arange(data_size)
        np.random.shuffle(indexes)
        return indexes

    def _to_tensor(self, data, transform_type):
        """
        transform data to tensor
        :param data:
        :param transform_type: tensor type
        :return:
        """
        return transform_type(data).to(self.device)

    def __next__(self):
        if self.index > self.n_batches or (self.index == self.n_batches and self._residue and self.drop_last):
            self.index = 0
            raise StopIteration
        else:
            start_ind, end_ind = self.index * self.batch_size, (self.index + 1) * self.batch_size
            if self.index == self.n_batches and self._residue:
                end_ind = self.data_size
            # 取和转换数据
            batches = []
            for i in range(self.batch_part):
                slice_data = self.batch_data[i][start_ind: end_ind]
                if self.to_tensor:
                    batches.append(self._to_tensor(slice_data, transform_type=self.tensor_type[i]))
                else:
                    batches.append(slice_data)
            batches = batches[0] if self.batch_part == 1 else batches
            self.index += 1
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self._residue:
            return self.n_batches + 1
        else:
            return self.n_batches

    def _check_to_tensor(self):
        if self.to_tensor:
            assert self.tensor_type is not None and \
                   isinstance(self.tensor_type, tuple) and len(self.tensor_type) == self.batch_part, \
                   "if to_tensor set True, you must specify tensor_type, and it's length must equal to batch_data"
            for ind, _type in enumerate(self.tensor_type):
                assert callable(_type), \
                    f"input tensor type {self.tensor_type[ind]} must be a valid and callable tensor type object"

    @staticmethod
    def _check_batch_data(batch_data):
        assert isinstance(batch_data, tuple) and len(batch_data) >= 1,\
            "batch_data must be tuple like object, like (x, ) or (x, y, z), etc"
        batch_part = len(batch_data)
        data_size = len(batch_data[0])
        if batch_part > 1:
            for ind, val in enumerate(batch_data):
                assert len(val) == data_size, \
                    f"the element of batch_data must have same length, batch_data[0] data_size is {data_size}, " \
                    f"but batch_data[{ind}] data_size is {len(val)}."
        return batch_data, batch_part, data_size


def self_iterator(batch_data, batch_size, shuffle=False, sampler=None, drop_last=False,
                  device=None, to_tensor=False, tensor_type=None):
    """
    根据输入的数据生成按照batch_size迭代器
    :param batch_data: tuple, 输入的数据，可以numpy格式，可以是tensor格式，如果一份数据，就是(x, )
    :param batch_size: int, 每次迭代取的数据样本大小
    :param shuffle: bool, 每次初始化迭代器时是否打乱数据，默认为False
    :param sampler: None or callable, 指定每次初始化迭代器时的采样策略
    :param drop_last: bool, 如果迭代最后剩余的数据不足一个batch，指定是否去掉，默认为True
    :param device: 如果转换数据为torch类型，指定GPU和CPU
    :param to_tensor: bool, 是否生成数据时转换为tensor类型
    :param tensor_type: tuple[Tensor.Type], 如果to_tensor为True，必须指定输入的每一部分batch_data的类型
    :return 迭代器
    """
    return SelfIterator(batch_data=batch_data, batch_size=batch_size, shuffle=shuffle,
                        sampler=sampler, drop_last=drop_last,
                        device=device, to_tensor=to_tensor, tensor_type=tensor_type)


# if __name__ == "__main__":
#     import torch
#     import numpy as np
#
#     x, y = np.random.rand(10, 3), np.random.rand(10)
#
#     test_iter = DataIterator(batch_data=(x, y), batch_size=3)
#     for x_, y_ in test_iter:
#         print(x_.shape, y_.shape)
#
#     # test drop_last
#     test_iter = DataIterator(batch_data=(x, y), batch_size=3, drop_last=True)
#     for x_, y_ in test_iter:
#         print(x_.shape, y_.shape)
#         print(type(x_), type(y_))
#
#     # transform tensor type
#     test_iter = DataIterator(batch_data=(x, y), batch_size=3, drop_last=True,
#                              to_tensor=True, tensor_type=(torch.Tensor, torch.LongTensor))
#     for x_, y_ in test_iter:
#         print(x_.shape, y_.shape)
#         print(type(x_), type(y_))
#
#     # test error
#     test_iter = DataIterator(batch_data=(x, y), batch_size=3, drop_last=True,
#                              to_tensor=True, tensor_type=(torch.Tensor, 2, ))
#     for x_, y_ in test_iter:
#         print(x_.shape, y_.shape)
#
#     # test error
#     y = np.random.rand(9)
#     test_iter = DataIterator(batch_data=(x, y), batch_size=3)
#
#     # test one
#     test_iter = DataIterator(batch_data=(x, ), batch_size=3)
#     for x_ in test_iter:
#         print(x_.shape)
#
#     # test shuffle
#     test_iter = DataIterator(batch_data=(x,), batch_size=3, shuffle=True)
#     for x_ in test_iter:
#         print(x_.shape)
#
#     # test sampler
#     def shuffle_sampler(batch_data):
#         data_size = len(batch_data[0])
#         indexes = np.arange(data_size)
#         np.random.shuffle(indexes)
#         return indexes
#     test_iter = DataIterator(batch_data=(x,), batch_size=3, sampler=shuffle_sampler)
#     for x_ in test_iter:
#         print(x_.shape)
