# -*- coding: utf-8 -*-

import gc
import pandas as pd
import numpy as np
from multiprocessing import Pool

from .model_utils import cross_validation_score, valid_set_score
from .spark_utils import load_driver_data
from .constants import JobType, ModelStage


def update_effect(old_effect, now_effect, min_err):
    if (min_err is not None and (now_effect - old_effect) > min_err) or \
            (min_err is None and now_effect > old_effect):
        return 0
    elif (min_err is not None and abs(now_effect - old_effect) <= min_err) or\
            min_err is None and now_effect == old_effect:
        return 1
    else:
        return 2


def get_effect(train_x, train_y, valid_x, valid_y, k_fold, model, model_param,
               model_fit_param, set_eval_set, metric_func, update_param_func,
               cross_val_score_params, random_state, feature_subset=None):
    if feature_subset is not None:
        train_x = train_x[:, feature_subset]
        valid_x = valid_x[:, feature_subset] if valid_x is not None else valid_x

    if k_fold is not None:
        best_effect, best_model_time = cross_validation_score(train_x, train_y, k_fold, model, model_param=model_param,
                                                              metric_func=metric_func, return_time=True,
                                                              random_state=random_state,
                                                              **cross_val_score_params)
    else:
        best_effect, best_model_time, bst_model = valid_set_score(train_x, train_y, valid_x, valid_y,
                                                                  model=model, model_param=model_param,
                                                                  model_fit_param=model_fit_param,
                                                                  metric_func=metric_func,
                                                                  return_model=True, return_time=True,
                                                                  set_eval_set=set_eval_set)
        if update_param_func is not None:
            model_param = update_param_func(model=bst_model, param=model_param)
    return best_effect, best_model_time, model_param


def initialize_effect(initialize_by_model, train_x, train_y, k_fold, model, model_param, valid_x, valid_y, metric_func,
                      model_fit_param, update_param_func, set_eval_set, cross_val_param, random_state, verbose):
    if initialize_by_model:
        effect_subset, subset_time, update_param = \
            get_effect(train_x, train_y, valid_x, valid_y, k_fold, model, model_param, model_fit_param,
                       set_eval_set, metric_func, update_param_func, cross_val_param, random_state, feature_subset=None)
        if verbose:
            print(f"initialize effect {effect_subset}, cost time {subset_time}, "
                  f"with feat_dim {train_x.shape[1]}, with param {update_param}")
    else:
        effect_subset, subset_time, update_param = float("-inf"), None, None
        if verbose:
            print(f"initialize effect {effect_subset}")
    return effect_subset, update_param, subset_time


def get_effect_print_news(ith, all_num, verbose, train_x, train_y, valid_x, valid_y, k_fold, model, param, model_fit_param,
                          set_eval_set, metric_func, update_param_func, cross_val_param, random_state, feature_subset):
    if verbose:
        print(f"round {ith}/{all_num} start...")
    effect_subset, subset_time, update_param = \
        get_effect(train_x, train_y, valid_x, valid_y, k_fold, model, param, model_fit_param,
                   set_eval_set, metric_func, update_param_func, cross_val_param, random_state, feature_subset)
    used_feat_dim = train_x.shape[1] if feature_subset is None else len(feature_subset)
    if verbose:
        print(f"round {ith}/{all_num} end, effect subset is {effect_subset}, "
              f"cost time {subset_time}, with feature dim is {used_feat_dim}, with param {update_param}")
    gc.collect()
    if feature_subset is None:
        return effect_subset, subset_time, update_param
    else:
        return effect_subset, subset_time, update_param, feature_subset


def local_loop_subsets(enable_multiprocess, n_jobs, subsets, verbose, train_x, train_y, valid_x,
                       valid_y, k_fold, model, model_fit_param, set_eval_set,
                       metric_func, update_param_func, cross_val_param, random_state, tag):
    assert tag in ("param_search", "feature_select"), "param tag should in ('param_search', 'feature_select')!"
    ret = []
    if enable_multiprocess:
        pool = Pool(processes=n_jobs)
        for index, subset in enumerate(subsets):
            if tag == "param_search":
                result = pool.apply_async(func=get_effect_print_news,
                                          args=(index + 1, len(subsets), verbose, train_x, train_y, valid_x,
                                                valid_y, k_fold, model, subset, model_fit_param, set_eval_set,
                                                metric_func, update_param_func, cross_val_param, random_state, None))
            else:
                result = pool.apply_async(func=get_effect_print_news,
                                          args=(index + 1, len(subsets), verbose, train_x, train_y, valid_x,
                                                valid_y, k_fold, model, None, model_fit_param, set_eval_set,
                                                metric_func, update_param_func, cross_val_param, random_state, subset))
            ret.append(result)
        pool.close()
        pool.join()
        ret = [i.get() for i in ret]
    else:
        for index, subset in enumerate(subsets):
            if tag == "param_search":
                result = get_effect_print_news(index + 1, len(subsets), verbose, train_x, train_y, valid_x,
                                               valid_y, k_fold, model, subset, model_fit_param, set_eval_set,
                                               metric_func, update_param_func, cross_val_param, random_state, None)
            else:
                result = get_effect_print_news(index + 1, len(subsets), verbose, train_x, train_y, valid_x,
                                               valid_y, k_fold, model, None, model_fit_param, set_eval_set,
                                               metric_func, update_param_func, cross_val_param, random_state, subset)
            ret.append(result)
    return ret


def spark_partition_loop(model, train_x, train_y, valid_x, valid_y, metric_func, subsets, k_fold, verbose,
                         hdfs_path, save_method, model_fit_param, update_param_func, set_eval_set, cross_val_param,
                         random_state, tag):
    assert tag in ("param_search", "feature_select"), "param tag should in ('param_search', 'feature_select')!"
    train_x, train_y, valid_x, valid_y = load_driver_data(hdfs_path, train_x, train_y, valid_x, valid_y, save_method)
    print("partition start...")
    print("train_x.shape", train_x.shape)
    print("train_y.shape", train_y.shape)
    subsets = [subset for _, subset in subsets]
    ret = []
    for index, subset in enumerate(subsets):
        if tag == "param_search":
            result = get_effect_print_news(index + 1, len(subsets), verbose, train_x, train_y, valid_x,
                                           valid_y, k_fold, model, subset, model_fit_param, set_eval_set,
                                           metric_func, update_param_func, cross_val_param, random_state, None)
        else:
            result = get_effect_print_news(index + 1, len(subsets), verbose, train_x, train_y, valid_x,
                                           valid_y, k_fold, model, None, model_fit_param, set_eval_set,
                                           metric_func, update_param_func, cross_val_param, random_state, subset)
        ret.append(result)
    print("partition end.")
    return ret


def check_X_y(X, y, stage, job, feature_info=None):
    assert X.ndim == 2, "input X's ndim must be 2!"
    if not isinstance(X, pd.DataFrame) and not isinstance(X, np.ndarray):
        raise ValueError("param must be pandas DataFrame or numpy.ndarray!")

    if stage == ModelStage.TRAIN:
        if y.ndim != 1 and not isinstance(y, pd.Series) and not isinstance(y, np.ndarray):
            raise ValueError("param must be pandas Series or numpy.ndarray and ndim == 1!")

        assert len(X) == len(y), "X'number must equal to y'number!"

        if job == JobType.CLASSIFICATION:
            y_unique_value = len(set(y))
            assert y_unique_value == 2, f"y's value contain {y_unique_value}, " \
                                        f"now classifier only support binary classification!"

        feature_info = {"feat_num": X.shape[1]}
        if isinstance(X, pd.DataFrame):
            feature_info["columns"] = list(X.columns)
        else:
            feature_info["columns"] = None
        return np.array(X), np.array(y), feature_info

    elif stage == ModelStage.PREDICT:
        assert X.shape[1] == feature_info["feat_num"], \
            f"input feature number {X.shape[1]} differ with train feature number {feature_info['feat_num']}"
        if isinstance(X, pd.DataFrame) and feature_info["columns"] is not None:
                for i, j in zip(list(X.columns), feature_info["columns"]):
                    if i != j:
                        raise ValueError(f"input feature name {i} differ with "
                                         f"train feature name {j} in same position!")
        return np.array(X)
    else:
        raise ValueError("now support input param is ('train', 'predict')!")


def get_n_random_list(alternative_list, fraction, n, random_state, bootstrap=False):
    """
    get n random fraction list from a list
    :param alternative_list:
    :param fraction:
    :param n:
    :param random_state:
    :param bootstrap
    :return: list[list]
    """
    length = len(alternative_list)
    if random_state is not None:
        np.random.seed(random_state)

    if not bootstrap and fraction == 1.0:
        return [np.array(range(length))]*n

    replace, select_num = (True, length) if bootstrap else (False, int(length*fraction))
    ret = []
    for _ in range(n):
        ret.append(np.random.choice(alternative_list, size=select_num, replace=replace))
    return ret
