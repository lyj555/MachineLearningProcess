# -*- coding: utf-8 -*-

import numpy as np
from torch.utils.data import Dataset, DataLoader


class TorchDataset(Dataset):
    def __init__(self, batch_data, data_size, to_tensor=False, tensor_type=None, device=None):
        self.batch_data = batch_data
        self.data_size = data_size
        self.to_tensor = to_tensor
        self.tensor_type = tensor_type
        self.device = device

    def __getitem__(self, index):
        if self.to_tensor:
            return tuple(self._to_tensor(np.array(part_data[index]), tensor_type)
                         for part_data, tensor_type in zip(self.batch_data, self.tensor_type))
        else:
            return tuple(np.array(i[index]) for i in self.batch_data)

    def _to_tensor(self, data, transform_type):
        """
        transform data to tensor
        :param data:
        :param transform_type: tensor type
        :return:
        """
        return transform_type(data).to(self.device)

    def __len__(self):
        return self.data_size


def _check_batch_data(batch_data):
    assert isinstance(batch_data, tuple) and len(batch_data) >= 1, \
        "batch_data must be tuple like object, like (x, ) or (x, y, z), etc"
    data_size = len(batch_data[0])
    batch_part = len(batch_data)
    if batch_part > 1:
        for ind, val in enumerate(batch_data):
            assert len(val) == data_size, \
                f"the element of batch_data must have same length, batch_data[0] data_size is {data_size}, " \
                f"but batch_data[{ind}] data_size is {len(val)}."
    return batch_data, batch_part, data_size


def _check_to_tensor(to_tensor, tensor_type, batch_part):
    if to_tensor:
        assert tensor_type is not None and \
               isinstance(tensor_type, tuple) and len(tensor_type) == batch_part, \
               "if to_tensor set True, you must specify tensor_type, and it's length must equal to batch_data"
        for ind, _type in enumerate(tensor_type):
            assert callable(_type), \
                f"input tensor type {tensor_type[ind]} must be a valid and callable tensor type object"


def torch_iterator(batch_data, batch_size=1, shuffle=False,
                   sampler=None, batch_sampler=None, num_workers=0, drop_last=False,
                   device=None, to_tensor=False, tensor_type=None, **kwargs):
    """
    根据输入的数据生成按照batch_size迭代器
    :param batch_data: tuple, 输入的数据，可以numpy格式，可以是tensor格式，如果一份数据，就是(x, )
    :param batch_size: int, 每次迭代取的数据样本大小
    :param shuffle: bool, 每次初始化迭代器时是否打乱数据，默认为False
    :param sampler: None or callable, 指定每次初始化迭代器时的采样策略
    :param batch_sampler: None or callable, 指定每个batch的样本，此时batch_size和shuffle参数失效
    :param num_workers: int, 指定加载数据时子进程的个数，默认为0
    :param drop_last: bool, 如果迭代最后剩余的数据不足一个batch，指定是否去掉，默认为True
    :param device: 如果转换数据为torch类型，指定GPU和CPU
    :param to_tensor: bool, 是否生成数据时转换为tensor类型
    :param tensor_type: tuple[Tensor.Type], 如果to_tensor为True，必须指定输入的每一部分batch_data的类型
    :param kwargs: torch.utils.data.DataLoader的其他参数
    :return: torch.utils.data.DataLoader的迭代器
    """
    batch_data, batch_part, data_size = _check_batch_data(batch_data)
    _check_to_tensor(to_tensor, tensor_type, batch_part)
    torch_dataset = TorchDataset(batch_data, data_size, to_tensor, tensor_type, device)
    return DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                      num_workers=num_workers, drop_last=drop_last, batch_sampler=batch_sampler, **kwargs)


# if __name__ == "__main__":
#     import torch
#     import numpy as np
#
#     x, y = np.random.rand(10, 3), np.random.rand(10)
#
#     test_iter = torch_iterator(batch_data=(x, y), batch_size=3)
#     for x_, y_ in test_iter:
#         print(x_.shape, y_.shape)
#
#     # test drop_last
#     test_iter = torch_iterator(batch_data=(x, y), batch_size=3, drop_last=True)
#     for x_, y_ in test_iter:
#         print(x_.shape, y_.shape)
#         print(type(x_), type(y_))
#
#     # transform tensor type
#     test_iter = torch_iterator(batch_data=(x, y), batch_size=3, drop_last=True,
#                                to_tensor=True, tensor_type=(torch.Tensor, torch.LongTensor))
#     for x_, y_ in test_iter:
#         print(x_.shape, y_.shape)
#         print(type(x_), type(y_))
#
#     # test error
#     test_iter = torch_iterator(batch_data=(x, y), batch_size=3, drop_last=True,
#                                to_tensor=True, tensor_type=(torch.Tensor, 2, ))
#     for x_, y_ in test_iter:
#         print(x_.shape, y_.shape)
#
#     # test error
#     y = np.random.rand(9)
#     test_iter = torch_iterator(batch_data=(x, y), batch_size=3)
#
#     # test one
#     test_iter = torch_iterator(batch_data=(x, ), batch_size=3)
#     for x_ in test_iter:
#         print(x_[0].shape)
#
#     # test shuffle
#     test_iter = torch_iterator(batch_data=(x,), batch_size=3, shuffle=True)
#     for x_ in test_iter:
#         print(x_[0].shape)
#
