# -*- coding: utf-8 -*-


def filter_feature_series(feat_series, threshold, max_features):
    if threshold is not None and (0 <= threshold <= 1):
        return feat_series[feat_series >= threshold].values.tolist()
    elif max_features is not None and (0 < max_features <= len(feat_series)):
        if not isinstance(max_features, int):
            max_features = int(len(feat_series)*max_features)
        return feat_series.sort_values(ascending=False)[:max_features].index.tolist()
    else:
        raise ValueError("param threshold and max_features set wrong.")
