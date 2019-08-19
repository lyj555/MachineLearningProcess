# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from utils.get_feature_by_contion import filter_feature_series


def null_filter(X, null_identity=None, max_features=None, return_indices=False, null_ratio_threshold=None):
    """
    filter features with max_features feature with not null
    :param X:
    :param null_identity: null的标识符,默认为None(np.NaN)
    :param null_ratio_threshold: null threshold for delete
    :param return_indices: False means return columns
    :param max_features: top [int or percent] features, only works if return_indices set False
    :return: list[str] or pandas.DataFrame
    """
    if null_identity is None:
        null_ratio = X.isnull().sum()*1.0/len(X)
    else:
        null_ratio = X.apply(lambda x: (x == null_identity).sum(), axis=0)*1.0/len(X)

    if return_indices:
        return null_ratio
    else:
        if 0 < null_ratio_threshold <= 1:
            return filter_feature_series(1-null_ratio, max_features=max_features, threshold=1-null_ratio_threshold)
        else:
            raise ValueError("null_ratio_threshold set wrong.")


def std_filter(X, max_features=None, null_ratio_threshold=None):
    """
    filter std, only process continuous variables
    :param X:
    :param max_features:
    :param null_ratio_threshold:
    :return:
    """
    continuous_var, dicrete_var = [], []


# def chi2(X, y, max_features, return_indices=False, data_types={}):
#     """
#     x2 test for X and y, only for continuous variable(greater than 0) and discrete label(classification task) works
#     :param X: pandas.DataFrame, feature matrix data
#     :param y: label data
#     :param return_indices: False means return
#     :param max_features: top [int or percent] features, only works if return_indices set False
#     :param data_types:
#     :return: list[str], feature list or pandas.DataFrame
#     """
#     X = X.astype(data_types).copy()



if __name__ == "__main__":
    # construct test data
    np.random.seed(666)
    data_size = 100
    df = pd.DataFrame({"f1": np.random.randint(1, 10, size=data_size),
                       "f2": np.random.rand(data_size),
                       "f3": np.random.choice(["A", "B", "C", "D"], size=data_size, replace=True),
                       "label_c": np.random.choice([0, 1], size=data_size, replace=True),
                       "label_f": np.random.rand(data_size)*10})
    print(df.head())
    ret = chi2(df[["f1", "f2"]], df["label_c"])
    print(chi2(df[["f1", "f2"]], df["label_c"]))