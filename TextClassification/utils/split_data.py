# -*- coding: utf-8 -*-

from sklearn.utils import shuffle


def split_data(data_list, split_ratio, random_state=None):
    assert sum(split_ratio) == 1, "split ratio sums must equal to 1"
    data_list = data_list.copy()

    data_list = shuffle(data_list, random_state=random_state)

    ret = []
    start_index, end_index = 0, None
    accum_ratio = 0
    for i in split_ratio:
        accum_ratio += i
        if end_index is None:
            end_index = int(len(data_list)*accum_ratio)
        else:
            start_index = end_index
            end_index = int(len(data_list)*accum_ratio)
        ret.append(data_list[start_index: end_index])
    return ret
