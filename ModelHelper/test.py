# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from model_helper.FeatureSelection.filter import null_filter, std_filter
from model_helper.FeatureSelection.wrapper import random_search, lvw, random_search_by_model_feat, top_feat_by_model

from model_helper.HyperparameterTuning.param_search import param_search
from model_helper.HyperparameterTuning.bayes_opt import bayes_search
from model_helper.HyperparameterTuning.hyper_opt import hyperopt_search


def make_data():
    data, targets = make_classification(
        n_samples=1000,
        n_features=50,
        n_informative=12,
        n_redundant=7,
        random_state=134985745,
    )
    cols = [f"feat{i}" for i in range(1, 1+data.shape[1])]
    data = pd.DataFrame(data, columns=cols)
    return data, targets


if __name__ == "__main__":
    # create data
    data, targets = make_data()

    # test feature selection wraper
    # clf = DecisionTreeClassifier()
    import lightgbm as lgb
    clf = lgb.LGBMClassifier()
    # 以全量特征作为初始化效果，随机选择80%的特征，进行比较筛选
    # feature_subset, subset_effect = random_search(data, targets, clf, initialize_by_model=True,
    #                                               k_fold=3, sample=0.8, max_iter=10, random_state=666)
    # 以全量特征作为初始化效果，随机选择n个特征，进行比较筛选通过metric_func
    # feature_subset, subset_effect = random_search(data, targets, clf, initialize_by_model=True,
    #                                               k_fold=3, sample=50, metric_func=roc_auc_score,
    #                                               max_iter=10, random_state=666,)
    # lvw
    # feature_subset, subset_effect = lvw(data, targets, clf, initialize_by_model=True,
    #                                     k_fold=None, create_valid=True, valid_ratio=0.2, metric_func=roc_auc_score,
    #                                     sample=0.8, max_iter=10, random_state=667)
    # random_search_by_model_feat
    # feature_subset, subset_effect = random_search_by_model_feat(data, targets, clf, initialize_by_model=True,
    #                                                             k_fold=None, create_valid=True, valid_ratio=0.2, metric_func=roc_auc_score,
    #                                                             sample=0.8, max_iter=10, random_state=667)
    # top_feat_by_model
    # feature_subset, subset_effect = top_feat_by_model(data, targets, clf, initialize_by_model=True,
    #                                                   top_ratio_list=[0.9, 0.8, 0.7], k_fold=None,
    #                                                   create_valid=True, valid_ratio=0.2, metric_func=roc_auc_score,
    #                                                   random_state=667)
    # hyper-parameter tuning
    # param_grid = {"max_depth": [1, 2, 3, 4, 5], "min_samples_leaf": [1, 10, 100, 200], "criterion": ["gini", "entropy"]}
    # best_effect, best_param = param_search(data, targets, clf, param_grid, method="random",
    #                                        k_fold=3, max_iter=20, random_state=666, create_valid=True, valid_ratio=0.2)

    # param_grid = {"max_depth": [1, 2, 3, 4, 5], "min_samples_leaf": [1, 10, 100, 200], "criterion": ["gini", "entropy"]}
    # best_effect, best_param = param_search(data, targets, clf, param_grid, method="grid",
    #                                        k_fold=3, random_state=666, create_valid=True, valid_ratio=0.2)
    # bayes opt
    # clf = RandomForestClassifier()
    # param_space = {"max_features": {"interval": (0.1, 0.9), "type": float},
    #                "n_estimators": {"interval": (10, 250), "type": int},
    #                "min_samples_split": {"interval": (2, 25), "type": int}
    #                }
    # best_result, best_params = bayes_search(data, targets, model=clf, param_space=param_space, n_iter=10,
    #                                         k_fold=3, random_state=666)
    # print(f"best_result is {best_result}, best_param is {best_params}")
    # hyper opt
    # clf = RandomForestClassifier()
    # param_space = {"max_features": hp.uniform("max_features", 0.1, 0.9),
    #                "n_estimators": hp.choice("n_estimators", range(10, 100)),
    #                "min_samples_split": hp.choice("min_samples_split", [2, 10, 100])
    #                }
    # trials, best_params = hyperopt_search(data, targets, model=clf, param_space=param_space, n_iter=10,
    #                                       k_fold=3, random_state=666)
    # for i in trials:
    #     print(i["result"]["stuff"])
    # print(f"best_param is {best_params}")

    from sklearn.metrics import log_loss
    metric_func = lambda y_true, y_pred: -log_loss(y_true, y_pred)
    feature_subset, subset_effect = random_search(data, targets, clf, initialize_by_model=True,
                                                  k_fold=None, sample=None, metric_func=metric_func,
                                                  create_valid=True, valid_ratio=0.2,
                                                  max_iter=10, random_state=666)
