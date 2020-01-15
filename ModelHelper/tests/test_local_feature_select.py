# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from model_helper.feature_selection.wrapper import random_search, lvw, weight_search, top_feat_search

np.random.seed(666)


if __name__ == "__main__":
    data_size = 1000
    feature_size = 100

    df = pd.DataFrame()
    for i in range(feature_size):
        df[f"f{i}"] = np.random.randint(i, i + 500, size=data_size)

    label = np.random.choice([0, 1], size=data_size, replace=True)

    print(df.head())

    print("test1 random_search k_fold...")
    clf = LGBMClassifier()
    subset, effect = random_search(df, label, clf, k_fold=3, sample=0.8, random_state=666, err_threshold=0.005)
    print(effect, subset)
    print("test1 done.")

    print("test2 random_search k_fold input is array")
    subset, effect = random_search(np.array(df), label, clf, k_fold=3, sample=0.8, random_state=666, err_threshold=0.005)
    print(effect, subset)
    print("test2 done.")

    print("test3 random_search create valid")
    subset, effect = random_search(df, label, clf, k_fold=None, sample=None, random_state=666,
                  create_valid=True, valid_ratio=0.2, metric_func=roc_auc_score)
    print(effect, subset)
    print("test3 done.")

    print("test4 self-defined valid set...")
    subset, effect = random_search(df, label, clf, sample=None, random_state=666,
                                   valid_x=df[:100], valid_y=label[:100], metric_func=roc_auc_score)
    print(effect, subset)
    print("test4 done.")

    print("test5 test add self-defined metric function...")
    subset, effect = random_search(df, label, clf, k_fold=None, sample=None, random_state=666,
                                   create_valid=True, valid_ratio=0.2,
                                   metric_func=lambda y_true, y_pred: roc_auc_score(y_true, y_pred))
    print(effect, subset)
    print("test5 done.")

    print("test6 valid_set_param...")

    def _update(model, param):
        if param is None and model.best_iteration_ is not None:
            return {"best_iteration": model.best_iteration_}
        elif param is not None:
            return param
        else:
            return None
    valid_set_param = {"model_fit_param": {"eval_metric": "auc", "verbose": False, "early_stopping_rounds": 5},
                       "set_eval_set": True,
                       "update_param_func": _update}
    clf = LGBMClassifier()
    subset, effect, param = random_search(df, label, clf, initialize_by_model=True, k_fold=None, sample=81, random_state=666,
                                          create_valid=True, valid_ratio=0.2, valid_set_param=valid_set_param,
                                          metric_func=roc_auc_score)
    print(effect, param, subset)
    print("test6 done.")

    print("test7 cross_val_param...")
    cross_val_param = {"scoring": lambda clf, X, y: roc_auc_score(y_true=y, y_score=clf.predict_proba(X)[:, 1]),
                       "n_jobs": None}
    subset, effect = random_search(df, label, clf, initialize_by_model=True, k_fold=None, sample=81, random_state=666,
                                   create_valid=True, valid_ratio=0.2, cross_val_param=cross_val_param,
                                   metric_func=roc_auc_score)
    print(effect, subset)
    print("test7 done.")

    print("test8 test multiprocess")
    clf = LGBMClassifier()
    subset, effect = random_search(df, label, clf, initialize_by_model=True, k_fold=3, sample=0.8, random_state=666,
                                   enable_multiprocess=True, n_jobs=2, err_threshold=0.005)
    print(effect, subset)
    print("test9 done.")

    print("test10 lvw k_fold")
    subset, effect = lvw(df, label, clf, k_fold=3, sample=0.8, random_state=667)
    print(subset, effect)
    print("test10 done.")

    print("test11 lvw create valid")
    subset, effect = lvw(df, label, clf, initialize_by_model=True, k_fold=None, sample=0.8, random_state=666,
                         create_valid=True, valid_ratio=0.2, metric_func=roc_auc_score)
    print(subset, effect)
    print("test11 done.")

    print("test12 lvw valid_set_param")

    def _update(model, param):
        if param is None and model.best_iteration_ is not None:
            return {"best_iteration": model.best_iteration_}
        elif param is not None:
            return param
        else:
            return None
    valid_set_param = {"model_fit_param": {"eval_metric": "auc", "verbose": False, "early_stopping_rounds": 1000},
                       "set_eval_set": True,
                       "update_param_func": _update}
    clf = LGBMClassifier()
    subset, effect, param = lvw(df, label, clf, initialize_by_model=True, k_fold=None, sample=81, random_state=666,
                                create_valid=True, valid_ratio=0.2, valid_set_param=valid_set_param,
                                metric_func=roc_auc_score)
    print(effect, param, subset)
    print("test12 done.")

    print("test13 model_feat k_fold...")
    clf = LGBMClassifier()
    a = weight_search(df, label, clf, k_fold=3, sample=0.8, random_state=667)
    print(a)
    print("test13 done.")

    print("test14 model_feat create valid...")
    a = weight_search(df, label, clf, initialize_by_model=True, k_fold=None, sample=0.8, random_state=667,
                      create_valid=True, valid_ratio=0.2, metric_func=roc_auc_score)
    print(a)
    print("test14 done")

    print("test15 model_feat min_feature...")
    a = weight_search(df, label, clf, initialize_by_model=True, k_fold=None, sample=None, random_state=667,
                      create_valid=True, valid_ratio=0.2, metric_func=roc_auc_score, min_feature=30)
    print(a)
    print("test15 done.")

    print("test16 model_feat valid_set_param...")

    def _update(model, param):
        if param is None and model.best_iteration_ is not None:
            return {"best_iteration": model.best_iteration_}
        elif param is not None:
            return param
        else:
            return None
    valid_set_param = {"model_fit_param": {"eval_metric": "auc", "verbose": False, "early_stopping_rounds": 1000},
                       "set_eval_set": True,
                       "update_param_func": _update}
    clf = LGBMClassifier()
    subset, effect, param = weight_search(df, label, clf, k_fold=None, sample=81, random_state=666,
                                          create_valid=True, valid_ratio=0.2, valid_set_param=valid_set_param,
                                          metric_func=roc_auc_score)
    print(effect, param, subset)
    print("test16 done.")

    print("test17 top_model_feat k_fold...")
    a = top_feat_search(df, label, top_ratio_list=[0.95, 0.9, 0.85, 0.8, 0.75, 0.7], initialize_by_model=True,
                        model=clf, k_fold=3, random_state=666)
    print(a)
    print("test17 done.")

    print("test18 top_model_feat create valid...")
    a = top_feat_search(df, label, top_ratio_list=[0.95, 0.9, 0.85, 0.8, 0.75, 0.7],
                        model=clf, k_fold=None, create_valid=True, valid_ratio=0.2, metric_func=roc_auc_score,
                        random_state=666)
    print(a)
    print("test18 done.")

    print("test19 top_model_feat valid_set_param...")

    def _update(model, param):
        if param is None and model.best_iteration_ is not None:
            return {"best_iteration": model.best_iteration_}
        elif param is not None:
            return param
        else:
            return None
    valid_set_param = {"model_fit_param": {"eval_metric": "auc", "verbose": False, "early_stopping_rounds": 1000},
                       "set_eval_set": True,
                       "update_param_func": _update}
    clf = LGBMClassifier()
    subset, effect, param = top_feat_search(df, label, clf, top_ratio_list=[0.95, 0.9, 0.85, 0.8, 0.75, 0.7],
                                            initialize_by_model=True, k_fold=None, random_state=666,
                                            create_valid=True, valid_ratio=0.2, valid_set_param=valid_set_param,
                                            metric_func=roc_auc_score)
    print(effect, param, subset)
    print("test19 done.")
