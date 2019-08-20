# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import train_test_split

from utils.feature_utils import generate_random_list
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


def random_search(train_x, train_y, k_fold=None,
                  create_valid=False, valid_ratio=None, valid_x=None, valid_y=None, metric_func=None,
                  sample=None, max_iter=10, model=None, random_state=None,
                  err_threshold=None, min_feature=None):
    """
    :param 
    """
    feat_dim = train_x.shape[1]

    if random_state is not None:
        np.random.seed(random_state)

    if min_feature is None:
        min_feature = 1
    elif 0 < min_feature < 1:
        min_feature = int(feat_dim*min_feature)
    elif 1 < min_feature < feat_dim:
        pass
    else:
        raise ValueError("min_feature set wrong!")
    
    best_effect, best_subset, best_feat_dim = float("-inf"), list(train_x.columns), feat_dim
    t = 0
    
    if create_valid and valid_ratio is not None and (0 < valid_ratio < 1):
        train_x, train_y, valid_x, valid_y = train_test_split(train_x, train_y, valid_ratio)
    
    while t < max_iter:
        feature_subset = generate_random_list(train_x.columns, sample, min_feature)
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


def lvw(train_x, train_y, k_fold=None,
        create_valid=False, valid_ratio=None, valid_x=None, valid_y=None, metric_func=None,
        sample=None, max_iter=10, model=None, random_state=None,
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
        if condition_num == 0 or condition_num == 1:
            best_effect, best_subset, best_feat_dim = effect_subset, feature_subset, feature_dim
            alternative_subset = feature_subset
            t = 0
        else:
            t += 1
    return best_subset


def random_search_by_model(train_x, train_y, k_fold=None,
        create_valid=False, valid_ratio=None, valid_x=None, valid_y=None, metric_func=None,
        sample=None, max_iter=10, model=None, random_state=None,
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
        if condition_num == 0 or condition_num == 1:
            best_effect, best_subset, best_feat_dim = effect_subset, feature_subset, feature_dim
            alternative_subset = feature_subset
            t = 0
        else:
            t += 1
    return best_subset
