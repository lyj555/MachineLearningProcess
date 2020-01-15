# -*- coding: utf-8 -*-

import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..utils.feature_utils import generate_random_list, generate_feature_list
from ..utils.common_utils import initialize_effect, update_effect, get_effect, \
    local_loop_subsets, spark_partition_loop
from ..utils.spark_utils import dynamic_confirm_partition_num, uniform_partition, save_driver_data


def _check_param(train_x, k_fold, metric_func, valid_set_param, cross_val_param, min_feature=None):
    if not isinstance(train_x, pd.DataFrame) and not isinstance(train_x, np.ndarray):
        raise ValueError("param train_x must be pandas DataFrame or numpy.ndarray")

    if k_fold is None and not callable(metric_func):
        raise ValueError("if k_fold set None, param metric_func must be callable object!")

    if valid_set_param is None:
        model_fit_param, update_param_func, set_eval_set = None, None, False
    else:
        model_fit_param = valid_set_param["model_fit_param"] if "model_fit_param" in valid_set_param else None
        update_param_func = valid_set_param["update_param_func"] if "update_param_func" in valid_set_param else None
        set_eval_set = valid_set_param["set_eval_set"] if "set_eval_set" in valid_set_param else None
    if cross_val_param is None:
        cross_val_param = {}

    if min_feature is not None:
        feat_dim = train_x.shape[1]
        if 0 < min_feature < 1:
            min_feature = int(feat_dim * min_feature)
        elif 1 <= min_feature < feat_dim:
            pass
        else:
            raise ValueError("min_feature set wrong!")
    return model_fit_param, update_param_func, set_eval_set, cross_val_param, min_feature


def _get_feature_importance(model, train_x, train_y, valid_x, valid_y, set_eval_set, model_fit_param):
    model = copy.deepcopy(model)
    if model_fit_param is None:
        clf = model.fit(train_x, train_y)
    else:
        if set_eval_set:
            clf = model.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], **model_fit_param)
        else:
            clf = model.fit(train_x, train_y, **model_fit_param)

    feat_imp = clf.feature_importances_

    feat_imp_sigmoid = 1 / (1 + np.exp(-feat_imp))
    feat_imp_sigmoid = feat_imp_sigmoid / sum(feat_imp_sigmoid)
    return feat_imp_sigmoid


def random_search(train_x, train_y, model, k_fold=None, create_valid=False, valid_ratio=None,
                  valid_x=None, valid_y=None, metric_func=None, sample=None, max_iter=10,
                  valid_set_param=None, cross_val_param=None,
                  enable_multiprocess=False, n_jobs=2,
                  random_state=None, initialize_by_model=True,
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
    :param max_iter: int, number iterations for feature subsets
    :param valid_set_param: dict or None, optional key is model_fit_param(dict, model fit param),
    update_param_func(callable function, with signature func(model, param), return param), the last output will
    return the param(for preparing the early_stopping_rounds, return the best_iteration), and set_eval_set(bool),
    if set True, it will add param eval_set=[(valid_x, valid_y)] in method model.fit, default is None
    :param cross_val_param: dict or None, sklearn's cross_val_score's param, default is None
    :param enable_multiprocess: bool, if not enable multiprocess
    :param n_jobs, int, the number of process, only valid when enable_multiprocess=True
    :param random_state: int or None, random seed
    :param err_threshold: float or None, if None, means if old_effect > best_effect then update best_effect = old_effect
    if float, means is (old_effect - best_effect) > err_threshold then update.
    :param min_feature: int float(0, 1) None, means minimum feature subset dim, default 1.
    :param verbose: bool, if print the effect of every round
    :return: tuple, (best_subset, best_effect) or (best_subset, best_effect, param) when valid_set_param is set
    """
    # check param
    model_fit_param, update_param_func, set_eval_set, cross_val_param, min_feature = \
        _check_param(train_x, k_fold, metric_func, valid_set_param, cross_val_param, min_feature)

    if random_state is not None:
        np.random.seed(random_state)

    cols = train_x.columns if isinstance(train_x, pd.DataFrame) else pd.Series(range(train_x.shape[1]))
    train_x, train_y = np.array(train_x), np.array(train_y)
    valid_x, valid_y = (np.array(valid_x), np.array(valid_y)) if valid_x is not None else (None, None)

    if create_valid and valid_ratio is not None and (0 < valid_ratio < 1):
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=valid_ratio,
                                                              random_state=random_state)

    best_effect, best_param, best_model_time = initialize_effect(initialize_by_model, train_x, train_y, k_fold,
                                                                 model, None, valid_x, valid_y,
                                                                 metric_func, model_fit_param, update_param_func,
                                                                 set_eval_set, cross_val_param, random_state, verbose)
    best_subset = np.array(range(train_x.shape[1]))
    feature_subsets = [generate_random_list(range(train_x.shape[1]), sample, min_feature) for _ in range(max_iter)]

    ret = local_loop_subsets(enable_multiprocess, n_jobs, feature_subsets, verbose, train_x, train_y, valid_x, valid_y,
                             k_fold, model, model_fit_param, set_eval_set, metric_func, update_param_func,
                             cross_val_param, random_state, tag="feature_select")

    for effect_subset, subset_time, update_param, feature_subset in ret:
        condition_num = update_effect(old_effect=best_effect, now_effect=effect_subset, min_err=err_threshold)
        if condition_num == 0 or (condition_num == 1 and len(feature_subset) < len(best_subset)):
            best_effect, best_subset, best_param = effect_subset, feature_subset, update_param
    if verbose:
        print("all round end.")
        print(f"best effect is {best_effect}, best feature dim {len(best_subset)}, input feat dim is {len(cols)}")
    if best_param is None:
        return list(cols[best_subset]), best_effect
    else:
        return list(cols[best_subset]), best_effect, best_param


def lvw(train_x, train_y, model, k_fold=None, create_valid=False, valid_ratio=None,
        valid_x=None, valid_y=None, metric_func=None, sample=None, max_iter=10,
        valid_set_param=None, cross_val_param=None,
        random_state=None, initialize_by_model=True,
        err_threshold=None, min_feature=1, verbose=True):
    """
    search feature subset by lvw(Las Vegas Wrapper), return the best
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
    :param max_iter: int, number iterations for feature subsets
    :param valid_set_param: dict or None, optional key is model_fit_param(dict, model fit param),
    update_param_func(callable function, with signature func(model, param), return param), the last output will
    return the param(for preparing the early_stopping_rounds, return the best_iteration), and set_eval_set(bool),
    if set True, it will add param eval_set=[(valid_x, valid_y)] in method model.fit, default is None
    :param cross_val_param: dict or None, sklearn's cross_val_score's param, default is None
    :param random_state: int or None, random seed
    :param err_threshold: float or None, if None, means if old_effect > best_effect then update best_effect = old_effect
    if float, means is (old_effect - best_effect) > err_threshold then update.
    :param min_feature: int float(0, 1) None, means minimum feature subset dim, default 1.
    :param verbose: bool, if print the effect of every round
    :return: tuple, (best_subset, best_effect) or (best_subset, best_effect, param) when valid_set_param is set
    """
    # check param
    model_fit_param, update_param_func, set_eval_set, cross_val_param, min_feature = \
        _check_param(train_x, k_fold, metric_func, valid_set_param, cross_val_param, min_feature)

    if random_state is not None:
        np.random.seed(random_state)

    cols = train_x.columns if isinstance(train_x, pd.DataFrame) else pd.Series(range(train_x.shape[1]))
    train_x, train_y = np.array(train_x), np.array(train_y)
    valid_x, valid_y = (np.array(valid_x), np.array(valid_y)) if valid_x is not None else (None, None)

    if create_valid and valid_ratio is not None and (0 < valid_ratio < 1):
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=valid_ratio,
                                                              random_state=random_state)

    best_effect, best_param, best_model_time = initialize_effect(initialize_by_model, train_x, train_y, k_fold,
                                                                 model, None, valid_x, valid_y,
                                                                 metric_func, model_fit_param, update_param_func,
                                                                 set_eval_set, cross_val_param, random_state, verbose)
    best_subset = np.array(range(train_x.shape[1]))

    t = 1
    alternative_subset = best_subset
    while t <= max_iter:
        if verbose:
            print(f"round {t} start...")
        feature_subset = generate_random_list(alternative_subset, sample, min_feature)
        effect_subset, subset_time, update_param = \
            get_effect(train_x, train_y, valid_x, valid_y, k_fold, model, None, model_fit_param,
                       set_eval_set, metric_func, update_param_func, cross_val_param, random_state, feature_subset=feature_subset)
        if verbose:
            print(f"effect subset is {effect_subset}, feature dim is {len(feature_subset)}, cost {subset_time} seconds.")
        condition_num = update_effect(old_effect=best_effect, now_effect=effect_subset, min_err=err_threshold)
        if condition_num == 0 or (condition_num == 1 and len(feature_subset) < len(best_subset)):
            best_effect, best_subset, best_param = effect_subset, feature_subset, update_param
            alternative_subset = feature_subset
            t = 0
        else:
            t += 1
    if verbose:
        print("all round end.")
        print(f"best effect is {best_effect}, best feature dim {len(best_subset)}, input feat dim is {len(cols)}")
    if best_param is None:
        return list(cols[best_subset]), best_effect
    else:
        return list(cols[best_subset]), best_effect, best_param


def weight_search(train_x, train_y, model, k_fold=None, create_valid=False, valid_ratio=None,
                  valid_x=None, valid_y=None, metric_func=None, sample=None, max_iter=10,
                  valid_set_param=None, cross_val_param=None,
                  enable_multiprocess=False, n_jobs=2,
                  random_state=None, initialize_by_model=True,
                  err_threshold=None, min_feature=1, verbose=True):
    """
    search feature subset(select feature by model feature importance) randomly, return the best
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
    :param max_iter: int, number iterations for feature subsets
    :param valid_set_param: dict or None, optional key is model_fit_param(dict, model fit param),
    update_param_func(callable function, with signature func(model, param), return param), the last output will
    return the param(for preparing the early_stopping_rounds, return the best_iteration), and set_eval_set(bool),
    if set True, it will add param eval_set=[(valid_x, valid_y)] in method model.fit, default is None
    :param cross_val_param: dict or None, sklearn's cross_val_score's param, default is None
    :param enable_multiprocess: bool, if not enable multiprocess
    :param n_jobs, int, the number of process, only valid when enable_multiprocess=True
    :param random_state: int or None, random seed
    :param err_threshold: float or None, if None, means if old_effect > best_effect then update best_effect = old_effect
    if float, means is (old_effect - best_effect) > err_threshold then update.
    :param min_feature: int float(0, 1) None, means minimum feature subset dim, default 1.
    :param verbose: bool, if print the effect of every round
    :return: tuple, (best_subset, best_effect) or (best_subset, best_effect, param) when valid_set_param is set
    """
    # check param
    model_fit_param, update_param_func, set_eval_set, cross_val_param, min_feature = \
        _check_param(train_x, k_fold, metric_func, valid_set_param, cross_val_param, min_feature)

    if random_state is not None:
        np.random.seed(random_state)

    cols = train_x.columns if isinstance(train_x, pd.DataFrame) else pd.Series(range(train_x.shape[1]))
    train_x, train_y = np.array(train_x), np.array(train_y)
    valid_x, valid_y = (np.array(valid_x), np.array(valid_y)) if valid_x is not None else (None, None)

    if create_valid and valid_ratio is not None and (0 < valid_ratio < 1):
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=valid_ratio,
                                                              random_state=random_state)

    best_effect, best_param, best_model_time = initialize_effect(initialize_by_model, train_x, train_y, k_fold,
                                                                 model, None, valid_x, valid_y,
                                                                 metric_func, model_fit_param, update_param_func,
                                                                 set_eval_set, cross_val_param, random_state, verbose)
    best_subset = np.array(range(train_x.shape[1]))

    feat_imp_sigmoid = _get_feature_importance(model, train_x, train_y, valid_x, valid_y, set_eval_set, model_fit_param)

    feature_subsets = [generate_random_list(range(train_x.shape[1]), sample, min_feature, p=feat_imp_sigmoid)
                       for _ in range(max_iter)]

    ret = local_loop_subsets(enable_multiprocess, n_jobs, feature_subsets, verbose, train_x, train_y, valid_x, valid_y,
                             k_fold, model, model_fit_param, set_eval_set, metric_func, update_param_func,
                             cross_val_param, random_state, tag="feature_select")

    for effect_subset, subset_time, update_param, feature_subset in ret:
        condition_num = update_effect(old_effect=best_effect, now_effect=effect_subset, min_err=err_threshold)
        if condition_num == 0 or (condition_num == 1 and len(feature_subset) < len(best_subset)):
            best_effect, best_subset, best_param = effect_subset, feature_subset, update_param
    if verbose:
        print("all round end.")
        print(f"best effect is {best_effect}, best feature dim {len(best_subset)}, input feat dim is {len(cols)}")
    if best_param is None:
        return list(cols[best_subset]), best_effect
    else:
        return list(cols[best_subset]), best_effect, best_param


def top_feat_search(train_x, train_y, model, top_ratio_list, k_fold=None, create_valid=False, valid_ratio=None,
                    valid_x=None, valid_y=None, metric_func=None,
                    valid_set_param=None, cross_val_param=None,
                    enable_multiprocess=False, n_jobs=2,
                    random_state=None, initialize_by_model=True,
                    err_threshold=None, verbose=True):
    """
    search feature subset by model feature importance top ratio list, return the best
    :param train_x: pandas.DataFrame, train set
    :param train_y: numpy.array or pd.Series, train label
    :param model: estimator, should contain method `fit` `predict_proba`
    :param top_ratio_list: list[float(0-1)], top ratio list feature by model's feature importance
    :param initialize_by_model: bool, initialize model effect by model, otherwise float("-inf"), default True
    :param k_fold: int or None, if not None, means use cross validation, default None
    :param create_valid: bool, if create valid set from param train_x, default False
    :param valid_ratio: float(0, 1) or None, the ratio of valid set, only works if param create_valid set True, default None
    :param valid_x: pandas.DataFrame or None, valid set, default None
    :param valid_y: numpy.array or pd.Series or None, valid set label, default None
    :param metric_func: function(y_true, y_pred) or None, if evaluate effect by valid set, should specify the metric function, default None
    :param valid_set_param: dict or None, optional key is model_fit_param(dict, model fit param),
    update_param_func(callable function, with signature func(model, param), return param), the last output will
    return the param(for preparing the early_stopping_rounds, return the best_iteration), and set_eval_set(bool),
    if set True, it will add param eval_set=[(valid_x, valid_y)] in method model.fit, default is None
    :param cross_val_param: dict or None, sklearn's cross_val_score's param, default is None
    :param enable_multiprocess: bool, if not enable multiprocess
    :param n_jobs, int, the number of process, only valid when enable_multiprocess=True
    :param random_state: int or None, random seed
    :param err_threshold: float or None, if None, means if old_effect > best_effect then update best_effect = old_effect
    if float, means is (old_effect - best_effect) > err_threshold then update.
    :param verbose: bool, if print the effect of every round
    :return: tuple, (best_subset, best_effect) or (best_subset, best_effect, param) when valid_set_param is set
    """
    # check param
    model_fit_param, update_param_func, set_eval_set, cross_val_param, min_feature = \
        _check_param(train_x, k_fold, metric_func, valid_set_param, cross_val_param, min_feature=None)

    if random_state is not None:
        np.random.seed(random_state)

    cols = train_x.columns if isinstance(train_x, pd.DataFrame) else pd.Series(range(train_x.shape[1]))
    train_x, train_y = np.array(train_x), np.array(train_y)
    valid_x, valid_y = (np.array(valid_x), np.array(valid_y)) if valid_x is not None else (None, None)

    if create_valid and valid_ratio is not None and (0 < valid_ratio < 1):
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=valid_ratio,
                                                              random_state=random_state)

    best_effect, best_param, best_model_time = initialize_effect(initialize_by_model, train_x, train_y, k_fold,
                                                                 model, None, valid_x, valid_y,
                                                                 metric_func, model_fit_param, update_param_func,
                                                                 set_eval_set, cross_val_param, random_state, verbose)
    best_subset = np.array(range(train_x.shape[1]))

    feat_imp = _get_feature_importance(model, train_x, train_y, valid_x, valid_y, set_eval_set, model_fit_param)
    feat_imp_series = pd.Series(feat_imp, index=list(range(train_x.shape[1])))
    alternative_list = generate_feature_list(feat_imp_series, top_ratio_list)

    ret = local_loop_subsets(enable_multiprocess, n_jobs, alternative_list, verbose, train_x, train_y, valid_x, valid_y,
                             k_fold, model, model_fit_param, set_eval_set, metric_func, update_param_func,
                             cross_val_param, random_state, tag="feature_select")

    for effect_subset, subset_time, update_param, feature_subset in ret:
        condition_num = update_effect(old_effect=best_effect, now_effect=effect_subset, min_err=err_threshold)
        if condition_num == 0 or (condition_num == 1 and len(feature_subset) < len(best_subset)):
            best_effect, best_subset, best_param = effect_subset, feature_subset, update_param
    if verbose:
        print("all round end.")
        print(f"best effect is {best_effect}, best feature dim {len(best_subset)}, input feat dim is {len(cols)}")
    if best_param is None:
        return list(cols[best_subset]), best_effect
    else:
        return list(cols[best_subset]), best_effect, best_param


def distributed_feature_select(spark, train_x, train_y, model, method="random", k_fold=None,
                               create_valid=False, valid_ratio=None, valid_x=None, valid_y=None,
                               metric_func=None, valid_set_param=None, cross_val_param=None,
                               sample=None, max_iter=10, random_state=None, top_ratio_list=None,
                               initialize_by_model=True, broadcast_variable=True, save_method="pandas",
                               num_partition=None, err_threshold=None, min_feature=1,
                               verbose=True):
    """
    search feature subset randomly, return the best
    :param spark: spark Session
    :param train_x: pandas.DataFrame, train set
    :param train_y: numpy.array or pd.Series, train label
    :param model: estimator, should contain method `fit` `predict_proba`
    :param method: str, ("random", "weight", "top_feat"), default is random
    :param k_fold: int or None, if not None, means use cross validation, default None
    :param create_valid: bool, if create valid set from param train_x, default False
    :param valid_ratio: float(0, 1) or None, the ratio of valid set, only works if param create_valid set True, default None
    :param valid_x: pandas.DataFrame or None, valid set, default None
    :param valid_y: numpy.array or pd.Series or None, valid set label, default None
    :param metric_func: function(y_true, y_pred) or None, if evaluate effect by valid set, should specify the metric function, default None
    :param valid_set_param: dict or None, optional key is model_fit_param(dict, model fit param),
    update_param_func(callable function, with signature func(model, param), return param), the last output will
    return the param(for preparing the early_stopping_rounds, return the best_iteration), and set_eval_set(bool),
    if set True, it will add param eval_set=[(valid_x, valid_y)] in method model.fit, default is None
    :param cross_val_param: dict or None, sklearn's cross_val_score's param, default is None
    :param sample: int or float or None, sample number, ratio, if set None, means
    :param max_iter: int, number iterations for feature subsets
    :param random_state: int or None, random seed
    :param top_ratio_list: list[float(0-1)], top ratio list feature by model's feature importance,
    only valid when param method="top_ratio"
    :param initialize_by_model: bool, initialize model effect by model, otherwise float("-inf"), default True
    :param num_partition: int or None, parallel partition number, default None, confirm by executor's num and core dynamically
    :param broadcast_variable: bool, whether or not create broadcast variable when send data to executor, default True
    :param save_method: str, ("pandas", "sparse", "numpy"), only valid when param broadcast_variable set False.
    :param err_threshold: float or None, if None, means if old_effect > best_effect then update best_effect = old_effect
    if float, means is (old_effect - best_effect) > err_threshold then update.
    :param min_feature: int float(0, 1) None, means minimum feature subset dim, default 1.
    :param verbose: bool, if print the effect of every round
    :return: tuple, (best_subset, best_effect) or (best_subset, best_effect, param) when valid_set_param is set
    """
    assert method in ("random", "weight", "top_feat"), 'method only support ("random", "weight", "top_feat")!'
    if method == "top_feat":
        assert top_ratio_list is not None, "when method set top_feat, param top_ratio list must not be None"
    # check param
    model_fit_param, update_param_func, set_eval_set, cross_val_param, min_feature = \
        _check_param(train_x, k_fold, metric_func, valid_set_param, cross_val_param, min_feature=min_feature)

    if random_state is not None:
        np.random.seed(random_state)

    cols = train_x.columns if isinstance(train_x, pd.DataFrame) else pd.Series(range(train_x.shape[1]))
    train_x, train_y = np.array(train_x), np.array(train_y)
    valid_x, valid_y = (np.array(valid_x), np.array(valid_y)) if valid_x is not None else (None, None)

    if create_valid and valid_ratio is not None and (0 < valid_ratio < 1):
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=valid_ratio,
                                                              random_state=random_state)

    best_effect, best_param, best_model_time = initialize_effect(initialize_by_model, train_x, train_y, k_fold,
                                                                 model, None, valid_x, valid_y,
                                                                 metric_func, model_fit_param, update_param_func,
                                                                 set_eval_set, cross_val_param, random_state, verbose)
    best_subset = np.array(range(train_x.shape[1]))

    cols_index = range(train_x.shape[1])
    feat_imp_sigmoid = None
    if method == "weight":
        feat_imp_sigmoid = _get_feature_importance(model, train_x, train_y, valid_x, valid_y, set_eval_set,
                                                   model_fit_param)
    if method in ("random", "weight"):
        feature_subsets = []
        for _ in range(max_iter):
            feat_subset = generate_random_list(cols_index, sample, min_num=min_feature, p=feat_imp_sigmoid)
            feature_subsets.append(feat_subset)
    else:  # top ratio list
        feat_imp = _get_feature_importance(model, train_x, train_y, valid_x, valid_y, set_eval_set, model_fit_param)
        feat_imp_series = pd.Series(feat_imp, index=list(cols_index))
        feature_subsets = generate_feature_list(feat_imp_series, top_ratio_list)

    if num_partition is None:
        num_partition = dynamic_confirm_partition_num(spark.sparkContext)
        print(f"the confirmed partition number is {num_partition}")
    else:
        print(f"the partition number is {num_partition} set by hand")
    s = uniform_partition(spark=spark, content_list=feature_subsets, num_partition=num_partition)

    train_x, train_y, valid_x, valid_y, hdfs_path_dic = \
        save_driver_data(spark, broadcast_variable, train_x, train_y, valid_x, valid_y, save_method)

    partition_result = s.mapPartitions(lambda x:
                                       spark_partition_loop(model, train_x, train_y, valid_x, valid_y,
                                                            metric_func, x, k_fold, verbose, hdfs_path=hdfs_path_dic,
                                                            save_method=save_method, model_fit_param=model_fit_param,
                                                            update_param_func=update_param_func,
                                                            set_eval_set=set_eval_set, cross_val_param=cross_val_param,
                                                            random_state=random_state, tag="feature_select")).collect()
    # find the best result
    for effect_subset, subset_time, param, feature_subset in partition_result:
        if verbose:
            print(f"effect subset is {effect_subset}, cost time {subset_time}, "
                  f"with feature num is {len(feature_subset)}")
        condition_num = update_effect(old_effect=best_effect, now_effect=effect_subset, min_err=err_threshold)
        if condition_num == 0 or (condition_num == 1 and len(feature_subset) < len(best_subset)):
            best_effect, best_subset, best_param = effect_subset, feature_subset, param
    if verbose:
        print(f"altogether {len(partition_result)} search rounds, search end.")
        print(f"best effect is {best_effect}, with best feature_dim {len(best_subset)}, input feat dim is {len(cols)}")

    if best_param is None:
        return list(cols[best_subset]), best_effect
    else:
        return list(cols[best_subset]), best_effect, best_param
