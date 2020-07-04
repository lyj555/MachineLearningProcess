# -*- coding: utf-8 -*-

import warnings
from collections import Iterable
import numpy as np


def _check_label(label):
    assert isinstance(label, (list, tuple, np.ndarray)) or isinstance(label, Iterable), \
        f"label must be list or tuple or numpy array"
    if isinstance(label, np.ndarray):
        assert label.ndim == 1, \
            f"if label's type is numpy.ndarray, its ndim must equal to 1, now it's {label.ndim}"
        return label
    return np.array(label)


def _check_sampler(labels, sampler):
    assert (isinstance(sampler, (int, float)) and sampler >= 0) or (isinstance(sampler, dict) and len(sampler) > 0), \
        "param sampler must be a number(>=0) or dict(length >= 0)"
    if isinstance(sampler, dict):
        ret = {}
        not_contain_labels = []
        for i in labels:
            if i not in sampler:
                not_contain_labels.append(i)
                ret[i] = 1.0
            else:
                assert isinstance(sampler[i], (int, float)) and sampler[i] >= 0, \
                    f"in param sampler, label {i} value must a number and >= 0"
                ret[i] = sampler[i]
        if not_contain_labels:
            warnings.warn(f"input label contain label value {not_contain_labels}, "
                          f"but sampler don't contain these labels, default keep all its index")
        not_contain_labels = []
        for i in sampler.keys():
            if i not in ret:
                not_contain_labels.append(i)
        if not_contain_labels:
            warnings.warn(f"input sampler key contain label {not_contain_labels}, "
                          f"the input data don't contain these labels!")
        return ret
    else:
        return {i: sampler for i in labels}


def __softmax(arr):
    ret = np.exp(arr)
    return ret / sum(ret)


def _check_sample_weight(sample_weight, length):
    assert sample_weight is None or length == len(sample_weight), \
        f"sample_weight must be None or it's length must be equal to label's length"
    if sample_weight is None:
        return sample_weight
    for i in sample_weight:
        assert i > 0, "sample_weight's all value must greater than 0"
    return sample_weight


def _select_sample(candidates, sample_ratio, candi_weight, random_seed):
    np.random.seed(random_seed)
    if 0 <= sample_ratio <= 1:
        replace = False  # 做无放回采样，相当于下采样
    else:
        replace = True  # 做有放回采样，相当于上采样
    sample_index = np.random.choice(candidates, size=int(len(candidates)*sample_ratio),
                                    replace=replace, p=candi_weight)
    return sample_index


def _select_index(arr, val):
    return np.where(arr == val)[0]


def sample_data_by_label(label, sampler, sample_weight=None, random_seed=None):
    """
    按照给定标签进行采样（上采样或者下采样），返回采样后的index。
    如果是上采样，会存在重复的index，下采样不会存在重复的index
    :param label: list or tuple or numpy.ndarray, 标签列，根据此列进行采样
    :param sampler: number or dict, 指定采样策略，
    如果是数字(0<=sampler<=1)，每个类别都会进行该比例的下采样；如果数字(sampler>1)，每个类别都会进行该比例的上采样。
    如果是字典，那么key值是类别，value是数字（如果小于1进行下采样，反之进行上采样），对于没有指定的标签，默认全部保留该标签的样本
    :param sample_weight: None or iterable, 指定每个样本的权重，采样时会根据此权重进行采样。
    :param random_seed: int or None，随机种子，复现采样结果。
    :return: numpy.ndarray
    """
    label = _check_label(label)
    sampler = _check_sampler(np.unique(label), sampler)
    sample_weight = _check_sample_weight(sample_weight, len(label))

    ret = np.array([], dtype=np.int32)
    for label_val, label_ratio in sampler.items():
        label_index = _select_index(label, label_val)
        candi_weight = None if sample_weight is None else __softmax(np.array(sample_weight)[label_index])
        indexes = _select_sample(label_index, sample_ratio=label_ratio,
                                 candi_weight=candi_weight, random_seed=random_seed)
        ret = np.append(ret, indexes)
    np.random.seed(random_seed)
    np.random.shuffle(ret)
    return ret


# if __name__ == "__main__":
#     t = np.array([0, 0, 0, 0, 1, 1, 1, 1])
#     sampler = 1.5
#     ind = sample_data_by_label(t, sampler)
#     print(ind)
#     print(t[ind], len(ind))
#
#     sampler = 0.5
#     ind = sample_data_by_label(t, sampler)
#     print(ind)
#     print(t[ind], len(ind))
#
#     sampler = {}
#     ind = sample_data_by_label(t, sampler)
#     print(ind)  # assert
#
#     sampler = {0: 0.5, 1: 2, 2: 1}
#     ind = sample_data_by_label(t, sampler)
#     print(ind)
#     print(t[ind], len(ind))
#
#     sampler = {0: 0.5, 2: 1}
#     ind = sample_data_by_label(t, sampler)
#     print(ind)
#     print(t[ind], len(ind))
#
#     # test sample_weight
#     t = np.array([0, 0, 0, 0, 1, 1, 1, 1])
#     sampler = 1.5
#     ind = sample_data_by_label(t, sampler, sample_weight=[1, 2])
#     print(ind)
#     print(t[ind], len(ind))  # assert
#
#     t = np.array([0, 0, 0, 0, 1, 1, 1, 1])
#     sampler = 1.5
#     ind = sample_data_by_label(t, sampler, sample_weight=[1, 2, 3, 4, 5, 6, 7, 8])
#     print(ind)
#     print(t[ind], len(ind))
#
#     t = np.array([0, 0, 0, 0, 1, 1, 1, 1])
#     sampler = 1.5
#     ind = sample_data_by_label(t, sampler, sample_weight=[1, 2, 3, 4, 5, 6, 7, 8], random_seed=234)
#     print(ind)
#     print(t[ind], len(ind))
#
#     t = np.random.choice([0, 1], 1000)
#     sampler = 0.5
#     ind = sample_data_by_label(t, sampler, random_seed=234)
#     print(ind)
#     print(t[ind], len(ind))
