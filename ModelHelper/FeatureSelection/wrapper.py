# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.feature_utils import generate_random_list, generate_feature_list
from utils.model_utils import cross_validation_score, valid_set_score


def _update_effect(old_effect, now_effect, min_err):
    if (min_err is not None and (now_effect - old_effect) > min_err) or \
            (min_err is None and now_effect > old_effect):
        return 0
    elif (min_err is not None and abs(now_effect - old_effect) <= min_err) or\
            min_err is None and now_effect == old_effect:
        return 1
    else:
        return 2


def _initialize(initialize_by_model, train_x, train_y, k_fold, model, valid_x, valid_y, metric_func, feat_dim):
    if initialize_by_model:
        if k_fold is not None:
            best_effect = cross_validation_score(train_x, train_y, k_fold, model)
        else:
            best_effect = valid_set_score(train_x, train_y, valid_x, valid_y,
                                          model=model, metric_func=metric_func)
        best_subset, best_feat_dim = list(train_x.columns), feat_dim
        print(f"initialize effect {best_effect}, feature dim {best_feat_dim}")
    else:
        best_effect, best_subset, best_feat_dim = float("-inf"), list(train_x.columns), feat_dim
        print(f"initialize effect -inf, feature dim {best_feat_dim}")
    return best_effect, best_subset, best_feat_dim


def random_search(train_x, train_y, model, initialize_by_model=True, k_fold=None,
                  create_valid=False, valid_ratio=None, valid_x=None, valid_y=None, metric_func=None,
                  sample=None, max_iter=10, random_state=None,
                  err_threshold=None, min_feature=1, verbose=True):
    """
    search feature subset randomly, return the best
    :param train_x: pandas.DataFrame, train set
    :param train_y: numpy.array or pd.Series, train label
    :param model: estimator, should contain method `fit` `predict_proba`
    :param initialize_by_model: bool, initialize model effect by model, otherwise float("-inf"), default True
    :param k_fold: int or None, if not None, means use cross validation, default None
    :param create_valid: bool, if create valid set from param train_x, default False
    :param valid_ratio: float(0, 1) or None, the ratio of valid set, only works if param create_valid set True, default None
    :param valid_x: pandas.DataFrame or None, valid set, default None
    :param valid_y: numpy.array or pd.Series or None, valid set label, default None
    :param metric_func: function(y_true, y_pred) or None, if evaluate effect by valid set, should specify the metric function, default None
    :param sample: int or float or None, sample number, ratio, if set None, means
    :param max_iter: int, number iterations for finding the answer
    :param random_state: int or None, random seed
    :param err_threshold: float or None, if None, means if old_effect > best_effect then update best_effect = old_effect
    if float, means is (old_effect - best_effect) > err_threshold then update.
    :param min_feature: int float(0, 1) None, means minimum feature subset dim, default 1.
    :param verbose: bool, if print the effect of every round
    :return: tuple, best_subset, best_feat_dim
    """
    if not isinstance(train_x, pd.DataFrame):
        raise ValueError("param train_x must be pandas DataFrame")
    feat_dim = train_x.shape[1]

    if random_state is not None:
        np.random.seed(random_state)

    if 0 < min_feature < 1:
        min_feature = int(feat_dim * min_feature)
    elif 1 <= min_feature < feat_dim:
        pass
    else:
        raise ValueError("min_feature set wrong!")

    if create_valid and valid_ratio is not None and (0 < valid_ratio < 1):
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=valid_ratio,
                                                              random_state=random_state)

    best_effect, best_subset, best_feat_dim = _initialize(initialize_by_model, train_x, train_y, k_fold,
                                                          model, valid_x, valid_y, metric_func, feat_dim)
    t = 1
    while t <= max_iter:
        if verbose:
            print(f"round {t} start...")
        feature_subset = generate_random_list(train_x.columns, sample, min_feature)
        # print(feature_subset)
        if feature_subset is None:
            return best_subset
        feature_dim = len(feature_subset)
        if k_fold is not None:
            effect_subset = cross_validation_score(train_x[feature_subset], train_y, k_fold, model)
        else:
            effect_subset = valid_set_score(train_x[feature_subset], train_y, valid_x[feature_subset], valid_y,
                                            model=model, metric_func=metric_func)
        if verbose:
            print(f"effect subset is {effect_subset}, feature dim is {feature_dim}")
        condition_num = _update_effect(old_effect=best_effect, now_effect=effect_subset, min_err=err_threshold)
        if condition_num == 0 or (condition_num == 1 and feature_dim < best_feat_dim):
            best_effect, best_subset, best_feat_dim = effect_subset, feature_subset, feature_dim
        t += 1
    print("all round end.")
    print(f"best effect is {best_effect}, best feature dim {best_feat_dim}")
    return best_subset, best_effect


def lvw(train_x, train_y, model, initialize_by_model=True, k_fold=None,
        create_valid=False, valid_ratio=None, valid_x=None, valid_y=None, metric_func=None,
        sample=None, max_iter=10, random_state=None,
        err_threshold=None, min_feature=1, verbose=True):
    if not isinstance(train_x, pd.DataFrame):
        raise ValueError("param train_x must be pandas DataFrame")
    feat_dim = train_x.shape[1]

    if random_state is not None:
        np.random.seed(random_state)

    if 0 < min_feature < 1:
        min_feature = int(feat_dim * min_feature)
    elif 1 <= min_feature < feat_dim:
        pass
    else:
        raise ValueError("min_feature set wrong!")

    if create_valid and valid_ratio is not None and (0 < valid_ratio < 1):
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=valid_ratio,
                                                              random_state=random_state)

    best_effect, best_subset, best_feat_dim = _initialize(initialize_by_model, train_x, train_y, k_fold,
                                                          model, valid_x, valid_y, metric_func, feat_dim)

    t = 1
    alternative_subset = best_subset
    while t <= max_iter:
        if verbose:
            print(f"round {t} start...")
        feature_subset = generate_random_list(alternative_subset, sample, min_feature)

        if feature_subset is None:
            return best_subset
        feature_dim = len(feature_subset)
        if k_fold is not None:
            effect_subset = cross_validation_score(train_x[feature_subset], train_y, k_fold, model)
        else:
            effect_subset = valid_set_score(train_x[feature_subset], train_y, valid_x[feature_subset], valid_y,
                                            model=model, metric_func=metric_func)
        if verbose:
            print(f"effect subset is {effect_subset}, feature dim is {feature_dim}")
        condition_num = _update_effect(old_effect=best_effect, now_effect=effect_subset, min_err=err_threshold)
        if condition_num == 0 or (condition_num == 1 and feature_dim < best_feat_dim):
            best_effect, best_subset, best_feat_dim = effect_subset, feature_subset, feature_dim
            alternative_subset = feature_subset
            t = 0
        else:
            t += 1
    print("all round end.")
    print(f"best effect is {best_effect}, best feature dim {best_feat_dim}")
    return best_subset, best_effect


def random_search_by_model_feat(train_x, train_y, model, initialize_by_model=True, k_fold=None,
                                create_valid=False, valid_ratio=None, valid_x=None, valid_y=None, metric_func=None,
                                sample=None, max_iter=10, random_state=None,
                                err_threshold=None, min_feature=1, verbose=True):
    if not isinstance(train_x, pd.DataFrame):
        raise ValueError("param train_x must be pandas DataFrame")
    feat_dim = train_x.shape[1]

    if random_state is not None:
        np.random.seed(random_state)

    if 0 < min_feature < 1:
        min_feature = int(feat_dim * min_feature)
    elif 1 <= min_feature < feat_dim:
        pass
    else:
        raise ValueError("min_feature set wrong!")

    if create_valid and valid_ratio is not None and (0 < valid_ratio < 1):
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=valid_ratio,
                                                              random_state=random_state)

    best_effect, best_subset, best_feat_dim = _initialize(initialize_by_model, train_x, train_y, k_fold,
                                                          model, valid_x, valid_y, metric_func, feat_dim)
    t = 1
    feat_imp = model.fit(train_x, train_y).feature_importances_

    feat_imp_sigmoid = 1/(1 + np.exp(-feat_imp))
    feat_imp_sigmoid = feat_imp_sigmoid/sum(feat_imp_sigmoid)
    # print("feature sigmoid", feat_imp_sigmoid)

    while t <= max_iter:
        if verbose:
            print(f"round {t} start...")
        feature_subset = generate_random_list(train_x.columns, sample, min_feature, p=feat_imp_sigmoid)

        if feature_subset is None:
            return best_subset
        feature_dim = len(feature_subset)

        if k_fold is not None:
            effect_subset = cross_validation_score(train_x[feature_subset], train_y, k_fold, model)
        else:
            effect_subset = valid_set_score(train_x[feature_subset], train_y, valid_x[feature_subset], valid_y,
                                            model=model, metric_func=metric_func)
        if verbose:
            print(f"effect subset is {effect_subset}, feature dim is {feature_dim}")

        condition_num = _update_effect(old_effect=best_effect, now_effect=effect_subset, min_err=err_threshold)
        # update best subset
        if condition_num == 0 or (condition_num == 1 and feature_dim < best_feat_dim):
            best_effect, best_subset, best_feat_dim = effect_subset, feature_subset, feature_dim
        t += 1
    print("all round end.")
    print(f"best effect is {best_effect}, best feature dim {best_feat_dim}")
    return best_subset, best_effect


def top_feat_by_model(train_x, train_y, model, top_ratio_list, initialize_by_model=True, k_fold=None,
                      create_valid=False, valid_ratio=None, valid_x=None, valid_y=None, metric_func=None,
                      random_state=None, err_threshold=None, verbose=True):
    if not isinstance(train_x, pd.DataFrame):
        raise ValueError("param train_x must be pandas DataFrame")

    feat_dim = train_x.shape[1]
    if random_state is not None:
        np.random.seed(random_state)

    if create_valid and valid_ratio is not None and (0 < valid_ratio < 1):
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=valid_ratio,
                                                              random_state=random_state)

    best_effect, best_subset, best_feat_dim = _initialize(initialize_by_model, train_x, train_y, k_fold,
                                                          model, valid_x, valid_y, metric_func, feat_dim)
    feat_imp = model.fit(train_x, train_y).feature_importances_
    feat_imp_series = pd.Series(feat_imp, index=train_x.columns.tolist())
    alternative_list = generate_feature_list(feat_imp_series, top_ratio_list)

    for ratio, feature_subset in zip(top_ratio_list, alternative_list):
        if verbose:
            print(f"ratio {ratio} start...")
        feature_dim = len(feature_subset)
        if k_fold is not None:
            effect_subset = cross_validation_score(train_x[feature_subset], train_y, k_fold, model)
        else:
            effect_subset = valid_set_score(train_x[feature_subset], train_y, valid_x[feature_subset], valid_y,
                                            model=model, metric_func=metric_func)
        if verbose:
            print(f"effect subset is {effect_subset}, feature dim is {feature_dim}")
        condition_num = _update_effect(old_effect=best_effect, now_effect=effect_subset, min_err=err_threshold)
        # update best effect
        if condition_num == 0 or (condition_num == 1 and feature_dim < best_feat_dim):
            best_effect, best_subset, best_feat_dim = effect_subset, feature_subset, feature_dim
    print("all round end.")
    print(f"best effect is {best_effect}, best feature dim {best_feat_dim}")
    return best_subset, best_effect


# if __name__ == "__main__":
#     import pandas as pd
#     import numpy as np
#
#     np.random.seed(666)
#     data_size = 1000
#     feature_size = 100
#
#     df = pd.DataFrame()
#     for i in range(feature_size):
#         df[f"f{i}"] = np.random.randint(i, i + 500, size=data_size)
#
#     label = np.random.choice([0, 1], size=data_size, replace=True)
#
#     print(df.head())
#
#     from sklearn.tree import DecisionTreeClassifier
#     from sklearn.metrics import roc_auc_score
#     from sklearn.model_selection import train_test_split
#
#     clf = DecisionTreeClassifier()

    # random_search(df, label, clf, initialize_by_model=True, k_fold=3, sample=0.8, random_state=666, err_threshold=0.005)

    # random_search(df, label, clf, initialize_by_model=True, k_fold=None, sample=81, random_state=666,
    #               create_valid=True, valid_ratio=0.2,
    #               metric_func=roc_auc_score)

    # lvw(df, label, clf, initialize_by_model=True, k_fold=3, sample=0.8, random_state=667)

    # lvw(df, label, clf, initialize_by_model=True, k_fold=None, sample=0.8, random_state=666,
    #     create_valid=True, valid_ratio=0.2, metric_func=roc_auc_score)

    # a = random_search_by_model_feat(df, label, clf, initialize_by_model=True, k_fold=3, sample=0.8, random_state=667)
    # print("last num", len(a))

    # random_search_by_model_feat(df, label, clf, initialize_by_model=True, k_fold=None, sample=0.8, random_state=667,
    #                                 create_valid=True, valid_ratio=0.2, metric_func=roc_auc_score)

    # random_search_by_model_feat(df, label, clf, initialize_by_model=True, k_fold=None, sample=None, random_state=667,
    #                             create_valid=True, valid_ratio=0.2, metric_func=roc_auc_score, min_feature=30)

    # top_feat_by_model(df, label, top_ratio_list=[0.95, 0.9, 0.85, 0.8, 0.75, 0.7], initialize_by_model=True,
    #                   model=clf, k_fold=3, random_state=666)

    # top_feat_by_model(df, label, top_ratio_list=[0.95, 0.9, 0.85, 0.8, 0.75, 0.7], initialize_by_model=True,
    #                   model=clf, k_fold=None, create_valid=True, valid_ratio=0.2, metric_func=roc_auc_score,
    #                   random_state=666)
