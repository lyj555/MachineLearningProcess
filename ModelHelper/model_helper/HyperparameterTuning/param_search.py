# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid

from ..utils.model_utils import cross_validation_score, valid_set_score


def _update_effect(old_effect, now_effect, min_err):
    if (min_err is not None and (now_effect - old_effect) > min_err) or \
            (min_err is None and now_effect > old_effect):
        return 0
    elif (min_err is not None and abs(now_effect - old_effect) <= min_err) or\
            min_err is None and now_effect == old_effect:
        return 1
    else:
        return 2


def _initialize(initialize_by_model, train_x, train_y, k_fold, model, valid_x, valid_y, metric_func):
    if initialize_by_model:
        if k_fold is not None:
            best_effect, best_model_time = cross_validation_score(train_x, train_y, k_fold, model,
                                                                  return_model_train_time=True)
        else:
            best_effect, best_model_time = valid_set_score(train_x, train_y, valid_x, valid_y, model=model,
                                                           metric_func=metric_func, return_model_train_time=True)
        # best_subset, best_feat_dim = list(train_x.columns), feat_dim
        print(f"initialize effect {best_effect}, cost time {best_model_time}, with param {model.get_params()}")
    else:
        best_effect = float("-inf")
        print(f"initialize effect -inf, cost time {best_model_time}, with param {model.get_params()}")
    return best_effect, model.get_params(), best_model_time


def _spark_partition_loop(train_x, train_y, model, valid_x, valid_y, metric_func, param_subset, k_fold, verbose):
    from pyspark.broadcast import Broadcast
    train_x = train_x.value if isinstance(train_x, Broadcast) else train_x
    train_y = train_y.value if isinstance(train_y, Broadcast) else train_y
    valid_x = valid_x.value if isinstance(valid_x, Broadcast) else valid_x
    valid_y = valid_y.value if isinstance(valid_y, Broadcast) else valid_y
    print("partition tune parameter start...")
    t = 1
    ret = []
    for _, param in param_subset:
        if verbose:
            print(f"round {t} start...")
        if k_fold is not None:
            effect_subset, subset_time = cross_validation_score(train_x, train_y, k_fold, model, model_param=param,
                                                                return_model_train_time=True)
        else:
            effect_subset, subset_time = valid_set_score(train_x, train_y, valid_x, valid_y, model=model,
                                                         metric_func=metric_func, model_param=param,
                                                         return_model_train_time=True)
        ret.append((effect_subset, param, subset_time))
        t += 1
    print("partition tune parameter end.")
    return ret


def _dynamic_confirm_partition_num(sc):
    driver_memory = sc.getConf().get('spark.driver.memory')
    driver_cores = sc.getConf().get('spark.driver.cores')

    if_dynamic_allocation = sc.getConf().get('spark.dynamicAllocation.enabled')
    min_executors = sc.getConf().get('spark.dynamicAllocation.minExecutors')
    max_executors = sc.getConf().get('spark.dynamicAllocation.maxExecutors')
    num_executors = sc.getConf().get('spark.executor.instances')
    executor_memory = sc.getConf().get('spark.executor.memory')
    executor_cores = sc.getConf().get('spark.executor.cores')
    print(f"driver info: driver memory is {driver_memory}, "
          f"driver core num is {driver_cores}")
    print(f"executor info: dynamic allocation set {if_dynamic_allocation},"
          f"min executors {min_executors}, max executors {max_executors},"
          f"num executors {num_executors}, executor memory {executor_memory}, executor cores {executor_cores}")
    if if_dynamic_allocation == "true":
        num_partitions = (int(max_executors)-1)*int(executor_cores)
    elif if_dynamic_allocation == "false":
        if num_executors is not None:
            num_partitions = int(num_executors)*int(executor_cores)
        else:
            num_partitions = 3  # default value
    else:
        num_partitions = 3  # default value
    return num_partitions


def param_search(train_x, train_y, model, param_grid, method="grid", initialize_by_model=True, k_fold=None, create_valid=False,
                 valid_ratio=None, valid_x=None, valid_y=None, metric_func=None, max_iter=10,
                 random_state=None, err_threshold=None, verbose=True):
    """
    parameter search with grid or random search
    :param train_x: pandas.DataFrame or numpy.ndarray, feature data
    :param train_y: pandas.DataFrame or pandas.Series or numpy.numpy.ndarray, one field, label data
    :param model: sklearn's estimator
    :param param_grid: parameter's search grid
    :param method: str, random or grid, parameter's search grid
    :param initialize_by_model: initialize the model effect, default True
    :param k_fold: use k_fold cv as evaluation score, None or int, default use cv, k_fold=5
    :param create_valid: bool or None, whether create valid set or not, default None
    :param valid_ratio: float (0,1), the ratio of valid set, only valid when create_valid True
    :param valid_x: pandas.DataFrame or numpy.ndarray or None, valid set, default None
    :param valid_y: pandas.DataFrame or numpy.ndarray or None, valid set, default None
    :param metric_func: function, measure y_true & y_pred, should only return one value
    :param max_iter: int or None, only valid when method chosen random
    :param random_state: int or None.
    :param err_threshold: None or float, min error between old_effect and now_effect
    :param verbose: bool or None, if print search results when search, default True
    :return: tuple, (best_effect, best_param)
    """
    if not isinstance(train_x, pd.DataFrame):
        raise ValueError("param train_x must be pandas DataFrame")
    if method not in ("grid", "random"):
        raise ValueError("param method must be in ('grid', 'random')!")

    if random_state is not None:
        np.random.seed(random_state)

    if create_valid and valid_ratio is not None and (0 < valid_ratio < 1):
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=valid_ratio,
                                                              random_state=random_state)

    best_effect, best_param, best_model_time = _initialize(initialize_by_model, train_x, train_y, k_fold,
                                                           model, valid_x, valid_y, metric_func)

    param_iter = ParameterGrid(param_grid)
    if method == "random":
        param_subset = np.random.choice(param_iter, size=(max_iter, ), replace=False)
    elif method == "grid":
        param_subset = param_iter

    t = 1
    for param in param_subset:
        if verbose:
            print(f"round {t} start...")

        if k_fold is not None:
            effect_subset, subset_time = cross_validation_score(train_x, train_y, k_fold, model, model_param=param,
                                                                return_model_train_time=True)
        else:
            effect_subset, subset_time = valid_set_score(train_x, train_y, valid_x, valid_y, model=model,
                                                         metric_func=metric_func, model_param=param,
                                                         return_model_train_time=True)
        if verbose:
            print(f"effect subset is {effect_subset}, cost time {subset_time}, with param {param}")
        condition_num = _update_effect(old_effect=best_effect, now_effect=effect_subset, min_err=err_threshold)
        if condition_num == 0 or (condition_num == 1 and subset_time < best_model_time):
            best_effect, best_param, best_model_time = effect_subset, param, subset_time
        t += 1
    print("all round end.")
    print(f"best effect is {best_effect}, cost time {best_model_time}, with best param {best_param}")
    return best_effect, best_param


def distributed_param_search(spark, train_x, train_y, model, param_grid, num_partition=None, method="grid",
                             initialize_by_model=True, k_fold=None, create_valid=False,
                             valid_ratio=None, valid_x=None, valid_y=None, metric_func=None,
                             max_iter=10, random_state=None, err_threshold=None, verbose=True):
    """
    distributed search parameter, only valid you have spark context
    :param spark: spark Session
    :param train_x: pandas.DataFrame or numpy.ndarray, feature data
    :param train_y: pandas.DataFrame or pandas.Series or numpy.numpy.ndarray, one field, label data
    :param model: sklearn's estimator
    :param param_grid: parameter's search grid
    :param num_partition: int or None, parallel partition number, default None, confirm by executor's num and core dynamically
    :param method: grid or random, default grid
    :param initialize_by_model: initialize the model effect, default True
    :param k_fold: use k_fold cv as evaluation score, None or int, default use cv, k_fold=5
    :param create_valid: bool or None, whether create valid set or not, default None
    :param valid_ratio: float (0,1), the ratio of valid set, only valid when create_valid True
    :param valid_x: pandas.DataFrame or numpy.ndarray, valid set
    :param valid_y: pandas.DataFrame or numpy.ndarray, valid set
    :param metric_func: function,
    :param max_iter: int, only valid when method chosen random
    :param random_state: int or None,
    :param err_threshold: min error between old_effect and now_effect
    :param verbose: bool or None, if print search results when search, default True
    :return: tuple, (best_effect, best_param)
    """
    if method not in ("grid", "random"):
        raise ValueError("param method must be in ('grid', 'random')!")

    if random_state is not None:
        np.random.seed(random_state)
    print(f"start parameter tuning, the input parameter grid is {param_grid}")

    if create_valid and valid_ratio is not None and (0 < valid_ratio < 1):
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=valid_ratio,
                                                              random_state=random_state)
    # create broadcast variable
    b_train_x = spark.sparkContext.broadcast(train_x) if train_x is not None else train_x
    b_train_y = spark.sparkContext.broadcast(train_y) if train_y is not None else train_y
    b_valid_x = spark.sparkContext.broadcast(valid_x) if valid_x is not None else valid_x
    b_valid_y = spark.sparkContext.broadcast(valid_y) if valid_y is not None else valid_y

    best_effect, best_param, best_model_time = _initialize(initialize_by_model, train_x, train_y, k_fold,
                                                           model, valid_x, valid_y, metric_func)
    param_iter = ParameterGrid(param_grid)
    if method == "random":
        param_subset = list(np.random.choice(param_iter, size=(max_iter, ), replace=False))
    elif method == "grid":
        param_subset = list(param_iter)
    np.random.shuffle(param_subset)

    if num_partition is None:
        num_partition = _dynamic_confirm_partition_num(spark.sparkContext)
        print(f"the confirmed partition number is {num_partition}")
    else:
        print(f"the partition number is {num_partition} set by hand")

    partitioner = [(i+1, param_subset[i]) for i in range(len(param_subset))]
    # para search
    s = spark.sparkContext.parallelize(partitioner, numSlices=1)
    s = s.partitionBy(num_partition, partitionFunc=lambda x: divmod(x, num_partition)[1])
    # check 是否均匀分区
    _ret = [len(i) for i in s.glom().collect()]
    print(f"max partition num {max(_ret)}, min partition_num {min(_ret)}")
    partition_result = s.mapPartitions(lambda x:
                                       _spark_partition_loop(b_train_x, b_train_y, model, b_valid_x, b_valid_y,
                                                             metric_func, x, k_fold, verbose)).collect()
    # find the best result
    t = 0
    for effect_subset, param, subset_time in partition_result:
        t += 1
        if verbose:
            print(f"effect subset is {effect_subset}, cost time {subset_time}, with param {param}")
        condition_num = _update_effect(old_effect=best_effect, now_effect=effect_subset, min_err=err_threshold)
        if condition_num == 0 or (condition_num == 1 and subset_time < best_model_time):
            best_effect, best_param, best_model_time = effect_subset, param, subset_time
    print(f"altogether {t} search rounds, search end.")
    print(f"best effect is {best_effect}, cost time {best_model_time}, with best param {best_param}")
    return best_effect, best_param


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
#     param_grid = {"max_depth": [1, 2, 3, 4, 5], "min_samples_leaf": [1, 10, 100, 200], "criterion": ["gini", "entropy"]}
#
#     # random_search(df, label, clf, param_grid, k_fold=3, max_iter=20, random_state=666)
#     # param_search(df, label, clf, param_grid, method="random",
#     #              k_fold=3, max_iter=20, random_state=666, create_valid=True, valid_ratio=0.2)
#
#     param_search(df, label, clf, param_grid, method="grid",
#                  k_fold=None, max_iter=20, random_state=666, create_valid=True, valid_ratio=0.2)
