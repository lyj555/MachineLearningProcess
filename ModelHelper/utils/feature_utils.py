# -*- coding: utf-8 -*-

import numpy as np


def filter_feature_series(feat_series, threshold, max_features, greater_than_and_equal=False):
    if threshold is not None and (0 <= threshold <= 1):
        if not greater_than_and_equal:
            return feat_series[feat_series > threshold].values.tolist()
        else:
            return feat_series[feat_series >= threshold].values.tolist()
    elif max_features is not None and (0 < max_features <= len(feat_series)):
        if not isinstance(max_features, int):
            max_features = int(len(feat_series)*max_features)
        return feat_series.sort_values(ascending=False)[:max_features].index.tolist()
    else:
        raise ValueError("param threshold and max_features set wrong.")


def generate_random_list(alternative_list, sample, min_num, p=None):
    length = len(alternative_list)
    if min_num >= length:
        return None

    if sample is None:
        sample_num = np.random.randint(min_num, length, 1)[0]
    elif 0 < sample < 1:
        sample_num = int(length * sample)
    elif sample >= 1:
        sample_num = sample

    if min_num >= sample_num:
        return None
    subset = np.random.choice(alternative_list, size=(sample_num, ), replace=False, p=p)
    return subset
