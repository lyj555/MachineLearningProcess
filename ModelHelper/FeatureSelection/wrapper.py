# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.feature_utils import generate_random_list, generate_feature_list
from utils.model_utils import cross_validation_score, valid_set_score


def _meet_condition(old_effect, now_effect, min_err):
    old_ret, old_dim = old_effect
    now_ret, now_dim = now_effect
    if (min_err is not None and (now_ret - old_ret) > min_err) or \
            (min_err is None and now_ret > old_ret):
        return 0
    elif (min_err is not None and abs(now_ret - old_ret) <= min_err) or\
            min_err is None and now_ret == old_ret:
        return 1
    else:
        return 2


def random_search(train_x, train_y, model, initialize_by_model=True, k_fold=None,
                  create_valid=False, valid_ratio=None, valid_x=None, valid_y=None, metric_func=None,
                  sample=None, max_iter=10, random_state=None,
                  err_threshold=None, min_feature=None):
    """
    search feature subset randomly, return the best
    :param train_x: train set
    :param train_y: train label
    :param model: sklearn estimator, should contain method `fit` `predict_proba`
    :param initialize_by_model: bool, initialize model effect by model, otherwise float("-inf"), default True
    :param k_fold: int or None, if not None, means use cross validation, default None
    :param create_valid:
    :param valid_ratio:
    :param valid_x:
    :param valid_y:
    :param metric_func:
    :param sample:
    :param max_iter:
    :param random_state:
    :param err_threshold:
    :param min_feature:
    :return:
    """
    if isinstance(train_x, pd.DataFrame):
        
    feat_dim = train_x.shape[1]

    if random_state is not None:
        np.random.seed(random_state)

    if min_feature is None:
        min_feature = 1
    elif 0 < min_feature < 1:
        min_feature = int(feat_dim * min_feature)
    elif 1 < min_feature < feat_dim:
        pass
    else:
        raise ValueError("min_feature set wrong!")

    if create_valid and valid_ratio is not None and (0 < valid_ratio < 1):
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=valid_ratio,
                                                              random_state=random_state)

    if initialize_by_model:
        if k_fold is not None:
            best_effect = cross_validation_score(train_x, train_y, k_fold, model)
            print("initialize:", best_effect)
        else:
            best_effect = valid_set_score(train_x, train_y, valid_x, valid_y,
                                          model=model, metric_func=metric_func)
            print("initialize:", best_effect)
        best_subset, best_feat_dim = list(train_x.columns), feat_dim
    else:
        best_effect, best_subset, best_feat_dim = float("-inf"), list(train_x.columns), feat_dim

    t = 1

    while t <= max_iter:
        print(f"round {t}...")
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
        print(effect_subset)
        condition_num = _meet_condition(old_effect=(best_effect, best_feat_dim), now_effect=(effect_subset, feature_dim)
                                        , min_err=err_threshold)
        if condition_num == 0:
            best_effect, best_subset, best_feat_dim = effect_subset, feature_subset, feature_dim
        elif condition_num == 1:
            if feature_dim < best_feat_dim:
                best_effect, best_subset, best_feat_dim = effect_subset, feature_subset, feature_dim
            else:
                pass
        else:
            pass
        t += 1
    return best_subset


def lvw(train_x, train_y, model, initialize_by_model=True, k_fold=None,
        create_valid=False, valid_ratio=None, valid_x=None, valid_y=None, metric_func=None,
        sample=None, max_iter=10, random_state=None,
        err_threshold=None, min_feature=None):
    feat_dim = train_x.shape[1]

    if random_state is not None:
        np.random.seed(random_state)

    if min_feature is None:
        min_feature = 1
    elif 0 < min_feature < 1:
        min_feature = int(feat_dim * min_feature)
    elif 1 < min_feature < feat_dim:
        pass
    else:
        raise ValueError("min_feature set wrong!")

    if initialize_by_model:
        if k_fold is not None:
            best_effect = cross_validation_score(train_x, train_y, k_fold, model)
        else:
            best_effect = valid_set_score(train_x, train_y, valid_x, valid_y,
                                          model=model, metric_func=metric_func)
        best_subset, best_feat_dim = list(train_x.columns), feat_dim
    else:
        best_effect, best_subset, best_feat_dim = float("-inf"), list(train_x.columns), feat_dim
    t = 0

    if create_valid and valid_ratio is not None and (0 < valid_ratio < 1):
        train_x, train_y, valid_x, valid_y = train_test_split(train_x, train_y, valid_ratio)

    alternative_subset = best_subset
    while t < max_iter:
        feature_subset = generate_random_list(alternative_subset, sample, min_feature)

        if feature_subset is None:
            return best_subset
        feature_dim = len(feature_subset)

        if k_fold is not None:
            effect_subset = cross_validation_score(train_x[feature_subset], train_y, k_fold, model)
        else:
            effect_subset = valid_set_score(train_x[feature_subset], train_y, valid_x[feature_subset], valid_y,
                                            model=model, metric_func=metric_func)

        condition_num = _meet_condition(old_effect=(best_effect, best_feat_dim), now_effect=(effect_subset, feature_dim)
                                        , min_err=err_threshold)
        if condition_num == 0 or (condition_num == 1 and feature_dim < best_feat_dim):
            best_effect, best_subset, best_feat_dim = effect_subset, feature_subset, feature_dim
            alternative_subset = feature_subset
            t = 0
        else:
            t += 1
    return best_subset


def random_search_by_model_feat(train_x, train_y, model, initialize_by_model=True, k_fold=None,
                                create_valid=False, valid_ratio=None, valid_x=None, valid_y=None, metric_func=None,
                                sample=None, max_iter=10, random_state=None,
                                err_threshold=None, min_feature=None):
    feat_dim = train_x.shape[1]

    if random_state is not None:
        np.random.seed(random_state)

    if min_feature is None:
        min_feature = 1
    elif 0 < min_feature < 1:
        min_feature = int(feat_dim * min_feature)
    elif 1 < min_feature < feat_dim:
        pass
    else:
        raise ValueError("min_feature set wrong!")

    if initialize_by_model:
        if k_fold is not None:
            best_effect = cross_validation_score(train_x, train_y, k_fold, model)
        else:
            best_effect = valid_set_score(train_x, train_y, valid_x, valid_y,
                                          model=model, metric_func=metric_func)
        best_subset, best_feat_dim = list(train_x.columns), feat_dim
    else:
        best_effect, best_subset, best_feat_dim = float("-inf"), list(train_x.columns), feat_dim
    t = 0

    if create_valid and valid_ratio is not None and (0 < valid_ratio < 1):
        train_x, train_y, valid_x, valid_y = train_test_split(train_x, train_y, valid_ratio)

    feat_imp = model.fit(train_x, train_y).feature_importances_

    feat_imp_sigmoid = 1/(1 + np.exp(-feat_imp))
    feat_imp_sigmoid = 1/sum(feat_imp_sigmoid)

    while t < max_iter:
        feature_subset = generate_random_list(train_x.columns, sample, min_feature, feat_imp_sigmoid)

        if feature_subset is None:
            return best_subset
        feature_dim = len(feature_subset)

        if k_fold is not None:
            effect_subset = cross_validation_score(train_x[feature_subset], train_y, k_fold, model)
        else:
            effect_subset = valid_set_score(train_x[feature_subset], train_y, valid_x[feature_subset], valid_y,
                                            model=model, metric_func=metric_func)

        condition_num = _meet_condition(old_effect=(best_effect, best_feat_dim), now_effect=(effect_subset, feature_dim)
                                        , min_err=err_threshold)
        # update best subset
        if condition_num == 0:
            best_effect, best_subset, best_feat_dim = effect_subset, feature_subset, feature_dim
        elif condition_num == 1:
            if feature_dim < best_feat_dim:
                best_effect, best_subset, best_feat_dim = effect_subset, feature_subset, feature_dim
            else:
                pass
        else:
            pass
    return best_subset


def top_feat_by_model(train_x, train_y, model, top_ratio_list, initialize_by_model=True, k_fold=None,
                      create_valid=False, valid_ratio=None, valid_x=None, valid_y=None, metric_func=None,
                      random_state=None, err_threshold=None):
    feat_dim = train_x.shape[1]

    if random_state is not None:
        np.random.seed(random_state)

    if initialize_by_model:
        if k_fold is not None:
            best_effect = cross_validation_score(train_x, train_y, k_fold, model)
        else:
            best_effect = valid_set_score(train_x, train_y, valid_x, valid_y,
                                          model=model, metric_func=metric_func)
        best_subset, best_feat_dim = list(train_x.columns), feat_dim
    else:
        best_effect, best_subset, best_feat_dim = float("-inf"), list(train_x.columns), feat_dim

    if create_valid and valid_ratio is not None and (0 < valid_ratio < 1):
        train_x, train_y, valid_x, valid_y = train_test_split(train_x, train_y, valid_ratio)

    feat_imp = model.fit(train_x, train_y).feature_importances_

    feat_imp_series = pd.Series(feat_imp, index=train_x.columns.tolist())
    alternative_list = generate_feature_list(feat_imp_series, top_ratio_list)

    for feature_subset in alternative_list:
        feature_dim = len(feature_subset)

        if k_fold is not None:
            effect_subset = cross_validation_score(train_x[feature_subset], train_y, k_fold, model)
        else:
            effect_subset = valid_set_score(train_x[feature_subset], train_y, valid_x[feature_subset], valid_y,
                                            model=model, metric_func=metric_func)

        condition_num = _meet_condition(old_effect=(best_effect, best_feat_dim), now_effect=(effect_subset, feature_dim)
                                        , min_err=err_threshold)
        if condition_num == 0 or (condition_num == 1 and feature_dim < best_feat_dim):
            best_effect, best_subset, best_feat_dim = effect_subset, feature_subset, feature_dim
    return best_subset


if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    np.random.seed(666)
    data_size = 1000
    feature_size = 100

    df = pd.DataFrame()
    for i in range(feature_size):
        df[f"f{i}"] = np.random.randint(i, i + 500, size=data_size)

    label = np.random.choice([0, 1], size=data_size, replace=True)

    print(df.head())

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    clf = DecisionTreeClassifier()

    # a = random_search(df, label, clf, initialize_by_model=True, k_fold=3, sample=0.8, random_state=666, err_threshold=0.005)
    # print(len(a))

    a = random_search(df, label, clf, initialize_by_model=True, k_fold=None, sample=81, random_state=666,
                      create_valid=True, valid_ratio=0.2,
                      metric_func=roc_auc_score)
    print("last num", len(a))

