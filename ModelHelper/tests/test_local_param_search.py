# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

from model_helper.hyper_parameter_tuning import param_search

np.random.seed(666)


if __name__ == "__main__":
    data_size = 1000
    feature_size = 100

    df = pd.DataFrame()
    for i in range(feature_size):
        df[f"f{i}"] = np.random.randint(i, i + 500, size=data_size)

    label = np.random.choice([0, 1], size=data_size, replace=True)
    print(df.head())

    clf = DecisionTreeClassifier()
    param_grid = {"max_depth": [1, 2, 3, 4, 5], "min_samples_leaf": [1, 10, 100, 200], "criterion": ["gini", "entropy"]}

    print("test1 k_fold...")
    a = param_search(df, label, clf, param_grid, k_fold=3, random_state=666)
    print(a)
    print("test1 done.")

    print("test2 create valid...")
    a = param_search(df, label, clf, param_grid, create_valid=True, valid_ratio=0.2, metric_func=roc_auc_score, random_state=666)
    print(a)
    print("test2 done.")

    print("test3 self-defined valid set...")
    a = param_search(df, label, clf, param_grid, valid_x=df[:100], valid_y=label[:100], metric_func=roc_auc_score, random_state=666)
    print(a)
    print("test3 done.")

    print("test4 test method random...")
    a = param_search(df, label, clf, param_grid, method="random", max_iter=10, k_fold=3,
                     random_state=666)
    print(a)
    print("test4 done.")

    print("test5 test add self-defined metric function...")
    a = param_search(df, label, clf, param_grid, create_valid=True, valid_ratio=0.2,
                     metric_func=lambda y_true, y_pred: roc_auc_score(y_true, y_pred),
                     random_state=666)
    print(a)
    print("test5 done.")

    print("test6 test valid_set_param...")

    def _update(model, param):
        if param is None:
            return model.get_params()
        else:
            param["n_estimators"] = model.best_iteration
        return param
    valid_set_param = {"model_fit_param": {"eval_metric": "auc", "verbose": False, "early_stopping_rounds": 5},
                       "set_eval_set": True,
                       "update_param_func": _update}
    clf = XGBClassifier()
    a = param_search(df, label, clf, param_grid, method="grid", random_state=666, create_valid=True,
                     valid_ratio=0.2, metric_func=roc_auc_score, valid_set_param=valid_set_param)
    print(a)
    print("test6 done.")

    print("test7 test cross_val_param...")
    cross_val_param = {"scoring": lambda clf, X, y: roc_auc_score(y_true=y, y_score=clf.predict_proba(X)[:, 1]),
                       "n_jobs": None}
    clf = XGBClassifier()
    a = param_search(df, label, clf, param_grid, method="grid", random_state=666, k_fold=5, cross_val_param=cross_val_param)
    print(a)
    print("test7 done.")

    print("test8 test multiprocess...")
    param_grid = {"max_depth": [1, 2, 3, 4, 5, 6], "min_samples_leaf": [1, 10, 100, 200], "criterion": ["gini", "entropy"]}
    from datetime import datetime
    s = datetime.now()
    param_search(df, label, clf, param_grid, method="grid", k_fold=5, random_state=666, metric_func=roc_auc_score)
    e = datetime.now()
    print(f"do not use multiprocess cost time {(e-s).seconds}")  # 147 seconds
    # usage 4
    s = datetime.now()
    param_search(df, label, clf, param_grid, method="grid", k_fold=5, random_state=666, metric_func=roc_auc_score,
                 enable_multiprocess=True, n_jobs=2)
    e = datetime.now()
    print(f"use multiprocess cost time {(e-s).seconds}")  # 40 seconds
    print("test8 done.")  # 77 seconds
