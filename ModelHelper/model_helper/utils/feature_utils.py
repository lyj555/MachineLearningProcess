# -*- coding: utf-8 -*-

import numpy as np


def filter_feature_series(feat_series, threshold, max_features, greater_than_and_equal=False):
    if threshold is not None and (0 <= threshold <= 1):
        if not greater_than_and_equal:
            return feat_series[feat_series > threshold].index.tolist()
        else:
            return feat_series[feat_series >= threshold].index.tolist()
    elif max_features is not None and (0 < max_features <= len(feat_series)):
        if not isinstance(max_features, int):
            max_features = int(len(feat_series)*max_features)
        return feat_series.sort_values(ascending=False)[:max_features].index.tolist()
    else:
        raise ValueError("param threshold and max_features set wrong.")


def generate_random_list(alternative_list, sample, min_num, p=None):
    length = len(alternative_list)
    assert length > min_num, f"the input list's length {length} should greater than min_num {min_num}!"

    if sample is None:
        sample_num = np.random.randint(min_num, length, 1)[0]
    elif 0 < sample < 1:
        sample_num = int(length * sample)
    elif sample >= 1:
        sample_num = sample

    if min_num > sample_num:
        raise ValueError(f"sample num {sample_num} should greater than min num {min_num}!")
    subset = np.random.choice(alternative_list, size=(sample_num, ), replace=False, p=p)
    return subset


def generate_feature_list(feature_series, top_list):
    feature_series = feature_series.sort_values(ascending=False)
    not_zero_feat = feature_series[feature_series != 0].index.tolist()
    not_zero_num = len(not_zero_feat)

    alternative_set = []
    for i in top_list:
        sample_num = int(not_zero_num*i)
        alternative_set.append(not_zero_feat[:sample_num])
    return alternative_set
