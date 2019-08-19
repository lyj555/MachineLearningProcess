# -*- coding: utf-8 -*-


from utils.get_feature_by_contion import filter_feature_series


def null_filter(X, null_identity=None, max_features=None, return_indices=False, null_ratio_threshold=None, greater_than_and_equal=False):
    """
    filter features with null values when null ratio greater than null_ratio_threshold or top max_feature not null feature
    :param X: pandas.DataFrame, feature data
    :param null_identity: identity of null value, default None(np.NaN)
    :param max_features: top [int or percent] features, only works if return_indices set False
    :param return_indices: False means return columns
    :param null_ratio_threshold: null threshold for delete
    :param greater_than_and_equal: default False, if contain operator euqual only works with null_ratio_threshold used
    :return: list[str] or pandas.Series
    """
    if null_identity is None:
        null_ratio = X.isnull().sum()*1.0/len(X)
    else:
        null_ratio = X.apply(lambda x: (x == null_identity).sum(), axis=0)*1.0/len(X)

    if return_indices:
        return null_ratio
    else:
        if null_ratio_threshold is not None and 0 < null_ratio_threshold <= 1:
            need_delete = filter_feature_series(null_ratio, max_features=None,
                                                threshold=null_ratio_threshold, greater_than_and_equal=greater_than_and_equal)
            return [i for i in X.columns if i not in need_delete]
        elif max_features is not None and max_features > 0:
            return filter_feature_series(1-null_ratio, max_features=max_features,
                                         threshold=None, greater_than_and_equal=greater_than_and_equal)
        else:
            raise ValueError("null_ratio_threshold set wrong.")


def std_filter(X, max_features=None, return_indices=False, std_threshold=None, greater_than_and_equal=None):
    """
    filter std, only process continuous variables(int, float, bool), keep object feature by default
    :param X: pandas.DataFrame, feature data
    :param max_features: top [int or percent] features, only works if return_indices set False
    :param return_indices: False means return columns
    :param std_threshold: default None, keep feature when its std greater than threshold
    :param greater_than_and_equal: default False, if contain operator euqual only works with std_threshold used
    :return: list[str] or pandas.Series
    """
    std_series = X.std()
    
    if return_indices:
        return std_series
    else:
        if std_threshold is not None and std_threshold >= 0:
            need_keep = filter_feature_series(std_series, max_features=None, 
                                              threshold=std_threshold, greater_than_and_equal=greater_than_and_equal)
            return [i for i in X.columns if i in need_keep or X[i].dtype == object]
        elif max_features is not None and max_features > 0:
            need_keep = filter_feature_series(std_series, max_features=max_features, 
                                              threshold=None, greater_than_and_equal=greater_than_and_equal)
            return [i for i in X.columns if i in need_keep or X[i].dtype == object]
        else:
            raise ValueError("null_ratio_threshold set wrong.")


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
    import pandas as pd
    import numpy as np
    
    np.random.seed(666)
    data_size = 100
    df = pd.DataFrame({"f1": np.random.randint(1, 10, size=data_size),
                       "f2": np.random.rand(data_size),
                       "f3": np.random.choice(["A", "B", "C", "D"], size=data_size, replace=True),
                       "f4": [10]*data_size, 
                       "label_c": np.random.choice([0, 1], size=data_size, replace=True),
                       "label_f": np.random.rand(data_size)*10})
    print(df.head())
    # ret = chi2(df[["f1", "f2"]], df["label_c"])
    # print(chi2(df[["f1", "f2"]], df["label_c"]))
    
