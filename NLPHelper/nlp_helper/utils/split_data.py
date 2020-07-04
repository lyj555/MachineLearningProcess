# -*- coding: utf-8 -*-

import numpy as np
from collections import Iterable


def _select_ele_by_ratios(candidates, split_ratios, label=None):
    candi_num, split_num = len(candidates), len(split_ratios)
    if label is None:
        assert_split_desc = f"sample number {candi_num} must greater than split part number {split_num}"
        assert_ratio_desc = "sample number {candi_num}, for split ratio {split_ratio}, select 0 elements."
    else:
        assert_split_desc = f"label={label} number is {candi_num}, must greater than {split_num}"
        assert_ratio_desc = "label={label} number is {candi_num}, for split ratio {split_ratio}, select 0 elements."
    assert len(candidates) > len(split_ratios), assert_split_desc

    # data_list = shuffle(data_list, random_state=random_state)

    ret = []
    start_index, end_index = None, 0
    accumulate_ratio = 0
    for i in split_ratios:
        accumulate_ratio += i
        start_index, end_index = end_index, int(candi_num * accumulate_ratio)
        if label is None:
            assert end_index - start_index > 0, assert_ratio_desc.format(candi_num=candi_num, split_ratio=i)
        else:
            assert end_index - start_index > 0, assert_ratio_desc.format(label=label, candi_num=candi_num, split_ratio=i)
        ret.append(candidates[start_index: end_index])
    return ret


def _check_indexes_and_label(indexes, by_label):
    assert isinstance(indexes, int) or isinstance(indexes, Iterable), \
        "param indexes must be a integer number or iterable object"
    if isinstance(indexes, int):
        indexes, n_sample = np.arange(indexes), indexes
    else:
        indexes = np.array(indexes)
        n_sample = len(indexes)
    assert n_sample >= 2, "index num must >= 2"
    assert by_label is None or len(by_label) == n_sample, \
        f"if by_label is not None, it's length must equal to indexes number."
    by_label = by_label if by_label is None else np.array(by_label)
    return indexes, n_sample, by_label


def _shuffle_data(random_seed, indexes, by_label):
    np.random.seed(seed=random_seed)
    shuffle_order = np.arange(len(indexes))
    np.random.shuffle(shuffle_order)

    indexes = indexes[shuffle_order]
    if by_label is not None:
        by_label = by_label[shuffle_order]
    return indexes, by_label


def split_data_with_index(indexes, split_ratios, by_label=None, random_seed=None):
    """
    指定切分的比例或者按照标签分布进行划分，返回每一部分的index
    :param indexes: int or indexes, 候选index集合，如果为int则默认从0到len(indexes)-1
    :param split_ratios: tuple, 指定切分的部分和每部分的切分比例，和需要为1，比如(0.2, 0.8)
    :param by_label: None or list, 指定标签列，会根据该列中标签比例进行切分，保证切分后的数据比例和此列基本一致
    :param random_seed: None or int, 随机种子，保证复现切分的结果
    :return: list[np.ndarray]
    """
    indexes, n_sample, by_label = _check_indexes_and_label(indexes, by_label)
    assert isinstance(split_ratios, (list, tuple)) and len(split_ratios) > 1 and sum(split_ratios) == 1, \
        "split ratio must be list or tuple and length greater than 1 and sums equal to 1"

    indexes, by_label = _shuffle_data(indexes=indexes, by_label=by_label, random_seed=random_seed)
    split_part = len(split_ratios)
    if by_label is None:
        assert split_part < n_sample, f"indexes num is {n_sample}, should greater than the split part {split_part}"
        candidates = [(indexes, None), ]  # for index
    else:
        candidates = [(indexes[np.array(by_label) == i], i) for i in np.unique(by_label)]  # find same label's indexes

    ret = None
    for candi, label in candidates:
        if ret is None:
            ret = _select_ele_by_ratios(candi, split_ratios=split_ratios, label=label)
        else:
            tmp = _select_ele_by_ratios(candi, split_ratios=split_ratios, label=label)
            for i in range(len(tmp)):
                ret[i] = np.append(ret[i], tmp[i])  # 纵向拼凑
    return ret


# if __name__ == "__main__":
#     t = split_data_with_index(10, split_ratios=(0.1, 0.1, 0.8))
#     print(t)
#     t = split_data_with_index(10, split_ratios=(0.1, 0.1, 0.8), random_seed=7)
#     print(t)
#     t = split_data_with_index(list(range(2, 12)), split_ratios=(0.1, 0.1, 0.8))
#     print(t)
#     t = split_data_with_index(2, split_ratios=(0.1, 0.1, 0.8))
#     print(t)  # assert
#     t = split_data_with_index(5, split_ratios=(0.1, 0.1, 0.8))
#     print(t)  # assert
#     t = split_data_with_index(20, split_ratios=(0.5, 0.5), by_label=[0,0,1,1,1,0,0,0,1,1]*2, random_seed=111)
#     print(t)
#
#     t = split_data_with_index(10020, split_ratios=(0.7, 0.1, 0.2))
#     print(len(t[0]), len(t[1]), len(t[2]))
#
    # sample_num = 5201
    # label = np.random.choice(["a", "b", "c", "d"], sample_num, p=[0.4, 0.1, 0.2, 0.3])
    # t = split_data_with_index(sample_num, split_ratios=(0.7, 0.1, 0.2), by_label=label, random_seed=222)
    # print(len(t[0]), len(t[1]), len(t[2]))
    # print(len(t[0]) + len(t[1]) + len(t[2]))
    # for i in t:
    #     print(len(i))
    #     a = label[i]
    #     print([round(sum(a==j)/len(i), 4) for j in ["a", "b", "c", "d"]])
