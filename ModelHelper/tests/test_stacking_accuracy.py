# -*- coding: utf-8 -*-

import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,\
    RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

from model_helper.model_ensemble import StackingClassifier, StackingRegressor

warnings.filterwarnings(action="ignore", category=FutureWarning)

if __name__ == "__main__":

    X, y = make_classification(n_samples=5000, n_features=20, n_classes=2, random_state=234)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)

    # test1 test normal process
    clf = StackingClassifier(base_learner_list=[RandomForestClassifier(), GradientBoostingClassifier(),
                                                DecisionTreeClassifier()],
                             meta_learner=LogisticRegression(), metric_func=roc_auc_score)
    clf.fit(X=train_x, y=train_y)

    pred = clf.predict_proba(X=test_x)[:, 1]
    auc_val = roc_auc_score(y_true=test_y, y_score=pred)
    print(auc_val)

    # test2 test model selector
    def selector(model_metrics):
        # print(model_metrics)
        model_avg_metric = np.array(list(map(lambda x: sum(x) / len(x), model_metrics)))
        return model_avg_metric.argsort()[-2:][::-1]  # get top 2 best model

    clf = StackingClassifier(base_learner_list=[RandomForestClassifier(), GradientBoostingClassifier(),
                                                DecisionTreeClassifier()],
                             meta_learner=LogisticRegression(), metric_func=roc_auc_score, select_base_learner=selector)
    clf.fit(X=train_x, y=train_y)

    pred = clf.predict_proba(X=test_x)[:, 1]
    auc_val = roc_auc_score(y_true=test_y, y_score=pred)
    print(auc_val)

    # test3 random select feature in base learner
    def selector(model_metrics):
        # print(model_metrics)
        model_avg_metric = np.array(list(map(lambda x: sum(x) / len(x), model_metrics)))
        return model_avg_metric.argsort()[-2:][::-1]  # get top 2 best model

    clf = StackingClassifier(base_learner_list=[RandomForestClassifier(), GradientBoostingClassifier(),
                                                DecisionTreeClassifier()],
                             meta_learner=LogisticRegression(), metric_func=roc_auc_score, select_base_learner=selector,
                             feature_fraction=0.8)
    clf.fit(X=train_x, y=train_y)

    pred = clf.predict_proba(X=test_x)[:, 1]
    auc_val = roc_auc_score(y_true=test_y, y_score=pred)
    print(auc_val)

    # test 4 multiprocessing
    def selector(model_metrics):
        # print(model_metrics)
        model_avg_metric = np.array(list(map(lambda x: sum(x) / len(x), model_metrics)))
        return model_avg_metric.argsort()[-2:][::-1]  # get top 2 best model

    clf = StackingClassifier(base_learner_list=[RandomForestClassifier(), GradientBoostingClassifier(),
                                                DecisionTreeClassifier()],
                             meta_learner=LogisticRegression(), metric_func=roc_auc_score, select_base_learner=selector,
                             feature_fraction=0.8, enable_multiprocess=True, n_jobs=2)
    clf.fit(X=train_x, y=train_y)

    pred = clf.predict_proba(X=test_x)[:, 1]
    auc_val = roc_auc_score(y_true=test_y, y_score=pred)
    print(auc_val)

    # test5 data validation X,y
    def selector(model_metrics):
        # print(model_metrics)
        model_avg_metric = np.array(list(map(lambda x: sum(x) / len(x), model_metrics)))
        return model_avg_metric.argsort()[-2:][::-1]  # get top 2 best model

    clf = StackingClassifier(base_learner_list=[RandomForestClassifier(), GradientBoostingClassifier(),
                                                DecisionTreeClassifier()],
                             meta_learner=LogisticRegression(), metric_func=roc_auc_score, select_base_learner=selector,
                             feature_fraction=0.8, enable_multiprocess=True, n_jobs=2)
    train_x = pd.DataFrame(train_x, columns=[f"f{i}" for i in range(train_x.shape[1])])
    clf.fit(X=train_x, y=train_y)

    pred = clf.predict_proba(test_x)[:, 1]  # train by DF, predict array, yes

    pred = clf.predict_proba(pd.DataFrame(test_x, columns=[f"f{i}" for i in range(test_x.shape[1])]))[:, 1]  # train by DF, predict DF

    # pred = clf.predict_proba(X=test_x[:, 2])[:, 1]  # wrong ndim
    # pred = clf.predict_proba(X=test_x[:, [0, 2]])[:, 1]  # wrong inpute feature dimension
    # test_df = pd.DataFrame(test_x, columns=[f"ff{i}" for i in range(test_x.shape[1])])
    # pred = clf.predict_proba(X=test_df)[:, 1]  # wrong feature name if use DF

    clf.fit(X=np.array(train_x), y=train_y)
    test_df = pd.DataFrame(test_x, columns=[f"ff{i}" for i in range(test_x.shape[1])])
    pred = clf.predict_proba(X=test_df)[:, 1]  # train by numpy array, predict DF
    auc_val = roc_auc_score(y_true=test_y, y_score=pred)

    # test stratify
    clf = StackingClassifier(base_learner_list=[RandomForestClassifier(), GradientBoostingClassifier(),
                                                DecisionTreeClassifier()],
                             meta_learner=LogisticRegression(), metric_func=roc_auc_score, select_base_learner=selector,
                             feature_fraction=0.8, enable_multiprocess=True, n_jobs=2)
    train_x = pd.DataFrame(train_x, columns=[f"f{i}" for i in range(train_x.shape[1])])
    clf.fit(X=train_x, y=train_y, stratify=True, stratify_col=train_y)

    pred = clf.predict_proba(test_x)[:, 1]  # train by DF, predict array, yes

    pred = clf.predict_proba(pd.DataFrame(test_x, columns=[f"f{i}" for i in range(test_x.shape[1])]))[:, 1]  # train by DF, predict DF
    auc_val = roc_auc_score(y_true=test_y, y_score=pred)
    print(auc_val)

    # test StackingRegressor
    X, y = make_regression(n_samples=5000, n_features=20, random_state=224)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)

    # test1 test normal process
    clf = StackingRegressor(base_learner_list=[RandomForestRegressor(), GradientBoostingRegressor(),
                                               DecisionTreeRegressor()],
                            meta_learner=LinearRegression())
    clf.fit(X=train_x, y=train_y)

    pred = clf.predict(X=test_x)
    r2_val = r2_score(y_true=test_y, y_pred=pred)
    print(r2_val)

    # test2 test model selector
    def selector(model_metrics):
        # print(model_metrics)
        model_avg_metric = np.array(list(map(lambda x: sum(x) / len(x), model_metrics)))
        return model_avg_metric.argsort()[:2]  # get top 2 best model

    clf = StackingRegressor(base_learner_list=[RandomForestRegressor(), GradientBoostingRegressor(),
                                               DecisionTreeRegressor()],
                            meta_learner=LinearRegression(),
                            metric_func=lambda y_true, y_score: r2_score(y_true=y_true, y_pred=y_score),
                            select_base_learner=selector)
    clf.fit(X=train_x, y=train_y)

    pred = clf.predict(X=test_x)
    r2_val = r2_score(y_true=test_y, y_pred=pred)
    print(r2_val)

    # test3 test feature fraction
    def selector(model_metrics):
        # print(model_metrics)
        model_avg_metric = np.array(list(map(lambda x: sum(x) / len(x), model_metrics)))
        return model_avg_metric.argsort()[:2]  # get top 2 best model

    clf = StackingRegressor(base_learner_list=[RandomForestRegressor(), GradientBoostingRegressor(),
                                               DecisionTreeRegressor()],
                            meta_learner=LinearRegression(),
                            metric_func=lambda y_true, y_score: r2_score(y_true=y_true, y_pred=y_score),
                            select_base_learner=selector, feature_fraction=0.8)
    clf.fit(X=train_x, y=train_y)

    pred = clf.predict(X=test_x)
    r2_val = r2_score(y_true=test_y, y_pred=pred)
    print(r2_val)

    # test4 multiprocessing
    def selector(model_metrics):
        # print(model_metrics)
        model_avg_metric = np.array(list(map(lambda x: sum(x) / len(x), model_metrics)))
        return model_avg_metric.argsort()[:2]  # get top 2 best model

    clf = StackingRegressor(base_learner_list=[RandomForestRegressor(), GradientBoostingRegressor(),
                                               DecisionTreeRegressor()],
                            meta_learner=LinearRegression(),
                            metric_func=lambda y_true, y_score: r2_score(y_true=y_true, y_pred=y_score),
                            select_base_learner=selector, feature_fraction=0.8, enable_multiprocess=True,
                            n_jobs=2)
    clf.fit(X=train_x, y=train_y)

    pred = clf.predict(X=test_x)
    r2_val = r2_score(y_true=test_y, y_pred=pred)
    print(r2_val)

