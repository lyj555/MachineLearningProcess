# -*- coding: utf-8 -*-

import numpy as np


def split_data(data_list, split_ratio, random_state=None):
    assert sum(split_ratio) == 1, "split ratio sums must equal to 1"
    data_list = data_list.copy()

    if random_state is not None:
        np.random.seed(random_state)
    np.random.shuffle(data_list)

    ret = []
    start_index, end_index = 0, None
    for i in split_ratio:
        if end_index is None:
            index_pair[1] = int(len(data_list)*i)

        ret.append(data_list[])




