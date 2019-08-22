# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid

from utils.model_utils import cross_validation_score, valid_set_score


def _update_effect(old_effect, now_effect, min_err):
    if (min_err is not None and (now_effect - old_effect) > min_err) or \
            (min_err is None and now_effect > old_effect):
        return 0
    elif (min_err is not None and abs(now_effect - old_effect) <= min_err) or\
            min_err is None and now_effect == now_effect:
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
        print(f"initialize effect {best_effect}, cost time {best_model_time}, with param {clf.get_params()}")
    else:
        best_effect = float("-inf")
        print(f"initialize effect -inf, cost time {best_model_time}, with param {clf.get_params()}")
    return best_effect, clf.get_params(), best_model_time


def random_search(train_x, train_y, model, param_grid, initialize_by_model=True, k_fold=None, create_valid=False,
                  valid_ratio=None, valid_x=None, valid_y=None, metric_func=None, max_iter=10,
                  random_state=None, err_threshold=None, verbose=True):
    if not isinstance(train_x, pd.DataFrame):
        raise ValueError("param train_x must be pandas DataFrame")

    if random_state is not None:
        np.random.seed(random_state)

    if create_valid and valid_ratio is not None and (0 < valid_ratio < 1):
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=valid_ratio,
                                                              random_state=random_state)

    best_effect, best_param, best_model_time = _initialize(initialize_by_model, train_x, train_y, k_fold,
                                                           model, valid_x, valid_y, metric_func)

    param_iter = ParameterGrid(param_grid)
    param_subset = np.random.choice(param_iter, size=(max_iter, ), replace=False)
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
    print(f"best effect is {best_effect}, cost time {subset_time}, with best param {best_param}")
    return best_param, best_effect


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
    param_grid = {"max_depth": [1, 2, 3, 4, 5], "min_samples_leaf": [1, 10, 100, 200], "criterion": ["gini", "entropy"]}

    # random_search(df, label, clf, param_grid, k_fold=3, max_iter=20, random_state=666)
    random_search(df, label, clf, param_grid, k_fold=3, max_iter=20, random_state=666, create_valid=True, valid_ratio=0.2)

