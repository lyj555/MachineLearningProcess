# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid

from ..utils.spark_utils import dynamic_confirm_partition_num, uniform_partition, \
    save_driver_data
from ..utils.common_utils import initialize_effect, update_effect, local_loop_subsets, spark_partition_loop


def _generate_param_combination(param_grid, method, max_iter):
    param_iter = ParameterGrid(param_grid)
    if method == "random":
        param_subset = np.random.choice(param_iter, size=(max_iter, ), replace=False)
    elif method == "grid":
        param_subset = list(param_iter)
    np.random.shuffle(param_subset)
    return param_subset


def param_search(train_x, train_y, model, param_grid, method="grid", k_fold=None,
                 create_valid=False, valid_ratio=None, valid_x=None, valid_y=None, metric_func=None, max_iter=10,
                 valid_set_param=None, cross_val_param=None, enable_multiprocess=False, n_jobs=2,
                 initialize_by_model=True, random_state=None, err_threshold=None, verbose=True):
    """
    parameter search with grid or random search
    :param train_x: pandas.DataFrame or numpy.ndarray, feature data
    :param train_y: pandas.DataFrame or pandas.Series or numpy.numpy.ndarray, one field, label data
    :param model: sklearn's estimator
    :param param_grid: parameter's search grid
    :param method: str, random or grid, parameter's search grid
    :param k_fold: use k_fold cv as evaluation score, None or int, default use cv, k_fold=5
    :param create_valid: bool or None, whether create valid set or not, default None
    :param valid_ratio: float (0,1), the ratio of valid set, only valid when create_valid True
    :param valid_x: pandas.DataFrame or numpy.ndarray or None, valid set, default None
    :param valid_y: pandas.DataFrame or numpy.ndarray or None, valid set, default None
    :param metric_func: callable or None, measure y_true & y_pred with signature(y_true, y_pred),
    should only return one value, used to measure the effect of valid set, default is None
    :param max_iter: int or None, only valid when method chosen random
    :param valid_set_param: dict or None, optional key is model_fit_param(dict, model fit param),
    update_param_func(callable function, with signature func(model, param), return param), the last output will
    return the param(for preparing the early_stopping_rounds, return the best_iteration), and set_eval_set(bool),
    if set True, it will add param eval_set=[(valid_x, valid_y)] in method model.fit, default is None
    :param cross_val_param: dict or None, sklearn's cross_val_score's param, default is None
    :param enable_multiprocess: bool, if not enable multiprocess
    :param n_jobs, int, the number of process, only valid when enable_multiprocess=True
    :param initialize_by_model: initialize the model effect, default True
    :param random_state: int or None.
    :param err_threshold: None or float, min error between old_effect and now_effect
    :param verbose: bool or None, if print search results when search, default True
    :return: tuple, (best_effect, best_param)
    """
    if not isinstance(train_x, pd.DataFrame) and not isinstance(train_x, np.ndarray):
        raise ValueError("param train_x must be pandas DataFrame or numpy.ndarray")
    if method not in ("grid", "random"):
        raise ValueError("param method must be in ('grid', 'random')!")
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

    if random_state is not None:
        np.random.seed(random_state)

    if k_fold is None and create_valid and valid_ratio is not None and (0 < valid_ratio < 1):
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=valid_ratio,
                                                              random_state=random_state)
    train_x, train_y, valid_x, valid_y = np.array(train_x), np.array(train_y), np.array(valid_x), np.array(valid_y)

    best_effect, best_param, best_model_time = initialize_effect(initialize_by_model, train_x, train_y, k_fold,
                                                                 model, model.get_params(), valid_x, valid_y,
                                                                 metric_func, model_fit_param, update_param_func,
                                                                 set_eval_set, cross_val_param, random_state, verbose)

    param_subset = _generate_param_combination(param_grid, method, max_iter)

    ret = local_loop_subsets(enable_multiprocess, n_jobs, param_subset, verbose, train_x, train_y, valid_x, valid_y,
                             k_fold, model, model_fit_param, set_eval_set, metric_func, update_param_func,
                             cross_val_param, random_state, tag="param_search")
    # loop for the best result
    for effect_subset, subset_time, update_param in ret:
        condition_num = update_effect(old_effect=best_effect, now_effect=effect_subset, min_err=err_threshold)
        if condition_num == 0 or (condition_num == 1 and subset_time < best_model_time):
            best_effect, best_param, best_model_time = effect_subset, update_param, subset_time
    if verbose:
        print("all round end.")
        print(f"best effect is {best_effect}, cost time {best_model_time}, with best param {best_param}")
    return best_effect, best_param


def distributed_param_search(spark, train_x, train_y, model, param_grid, method="grid",
                             k_fold=None, create_valid=False, valid_ratio=None, valid_x=None, valid_y=None,
                             metric_func=None, max_iter=10, random_state=None,
                             valid_set_param=None, cross_val_param=None, initialize_by_model=True,
                             num_partition=None, broadcast_variable=True, save_method="pandas",
                             err_threshold=None, verbose=True):
    """
    distributed search parameter, only valid you have spark context
    :param spark: spark Session
    :param train_x: pandas.DataFrame or numpy.ndarray, feature data
    :param train_y: pandas.DataFrame or pandas.Series or numpy.numpy.ndarray, one field, label data
    :param model: sklearn's estimator
    :param param_grid: dict, parameter's search grid
    :param method: str, grid or random, default grid
    :param k_fold: use k_fold cv as evaluation score, None or int, default use cv, k_fold=5
    :param create_valid: bool or None, whether create valid set or not, default None
    :param valid_ratio: float (0,1), the ratio of valid set, only valid when create_valid True
    :param valid_x: pandas.DataFrame or numpy.ndarray, valid set
    :param valid_y: pandas.DataFrame or numpy.ndarray, valid set
    :param metric_func: callable or None, measure y_true & y_pred with signature(y_true, y_pred),
    should only return one value, used to measure the effect of valid set, default is None
    :param max_iter: int, only valid when method chosen random
    :param random_state: int or None,
    :param valid_set_param: dict or None, optional key is model_fit_param(dict, model fit param),
    update_param_func(callable function, with signature func(model, param), return param), the last output will
    return the param(for preparing the early_stopping_rounds, return the best_iteration), and set_eval_set(bool),
    if set True, it will add param eval_set=[(valid_x, valid_y)] in method model.fit, default is None
    :param cross_val_param: dict or None, sklearn's cross_val_score's param, default is None
    :param num_partition: int or None, parallel partition number, default None, confirm by executor's num and core dynamically
    :param broadcast_variable: bool, whether or not create broadcast variable when send data to executor, default True
    :param save_method: str, ("pandas", "sparse", "numpy"), only valid when param broadcast_variable set False.
    :param initialize_by_model: initialize the model effect, default True
    :param err_threshold: min error between old_effect and now_effect
    :param verbose: bool or None, if print search results when search, default True
    :return: tuple, (best_effect, best_param)
    """
    if not isinstance(train_x, pd.DataFrame) and not isinstance(train_x, np.ndarray):
        raise ValueError("param train_x must be pandas DataFrame or numpy.ndarray")
    if method not in ("grid", "random"):
        raise ValueError("param method must be in ('grid', 'random')!")
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

    if not broadcast_variable:
        assert save_method in ("pandas", "sparse", "numpy"), "when param broadcast_variable set False, " \
                                                             "save_method should in ('pandas', 'sparse', 'numpy')"

    if random_state is not None:
        np.random.seed(random_state)
    print(f"start parameter tuning, the input parameter grid is {param_grid}")

    if k_fold is None and create_valid and valid_ratio is not None and (0 < valid_ratio < 1):
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=valid_ratio,
                                                              random_state=random_state)

    train_x, train_y = np.array(train_x), np.array(train_y)
    valid_x, valid_y = (np.array(valid_x), np.array(valid_y)) if valid_x is not None else (None, None)

    best_effect, best_param, best_model_time = initialize_effect(initialize_by_model, train_x, train_y, k_fold,
                                                                 model, model.get_params(), valid_x, valid_y,
                                                                 metric_func, model_fit_param, update_param_func,
                                                                 set_eval_set, cross_val_param, random_state, verbose)
    param_subset = _generate_param_combination(param_grid, method, max_iter)

    if num_partition is None:
        num_partition = dynamic_confirm_partition_num(spark.sparkContext)
        print(f"the confirmed partition number is {num_partition}")
    else:
        print(f"the partition number is {num_partition} set by hand")

    s = uniform_partition(spark=spark, content_list=param_subset, num_partition=num_partition)

    train_x, train_y, valid_x, valid_y, hdfs_path_dic = \
        save_driver_data(spark, broadcast_variable, train_x, train_y, valid_x, valid_y, save_method)

    partition_result = s.mapPartitions(lambda x:
                                       spark_partition_loop(model, train_x, train_y, valid_x, valid_y,
                                                            metric_func, x, k_fold, verbose, hdfs_path=hdfs_path_dic,
                                                            save_method=save_method, model_fit_param=model_fit_param,
                                                            update_param_func=update_param_func,
                                                            set_eval_set=set_eval_set, cross_val_param=cross_val_param,
                                                            random_state=random_state, tag="param_search")).collect()
    # loop and find the best result
    for effect_subset, subset_time, param in partition_result:
        if verbose:
            print(f"effect subset is {effect_subset}, cost time {subset_time}, with param {param}")
        condition_num = update_effect(old_effect=best_effect, now_effect=effect_subset, min_err=err_threshold)
        if condition_num == 0 or (condition_num == 1 and subset_time < best_model_time):
            best_effect, best_param, best_model_time = effect_subset, param, subset_time
    if verbose:
        print(f"altogether {len(partition_result)} search rounds, search end.")
        print(f"best effect is {best_effect}, cost time {best_model_time}, with best param {best_param}")
    return best_effect, best_param
