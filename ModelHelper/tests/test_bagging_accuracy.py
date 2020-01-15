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
# from sklearn.linear_model import LogisticRegression, LinearRegression

from model_helper.model_ensemble import BaggingClassifier, BaggingRegressor

warnings.filterwarnings(action="ignore", category=FutureWarning)

if __name__ == "__main__":

    X, y = make_classification(n_samples=5000, n_features=20, n_classes=2, random_state=234)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)

    # test1 test normal process
    clf = BaggingClassifier(base_learner_list=[RandomForestClassifier(), GradientBoostingClassifier(),
                                               DecisionTreeClassifier()])
    clf.fit(X=train_x, y=train_y)

    pred = clf.predict_proba(X=test_x)[:, 1]
    auc_val = roc_auc_score(y_true=test_y, y_score=pred)
    print(auc_val)

    # test2 test feature fraction
    clf = BaggingClassifier(base_learner_list=[RandomForestClassifier(), GradientBoostingClassifier(),
                                               DecisionTreeClassifier()],
                            feature_fraction=0.8)
    clf.fit(X=train_x, y=train_y)
    print(clf._feature_indexer)

    pred = clf.predict_proba(X=test_x)[:, 1]
    auc_val = roc_auc_score(y_true=test_y, y_score=pred)
    print(auc_val)

    # test3 bootstrap
    clf = BaggingClassifier(base_learner_list=[RandomForestClassifier(), GradientBoostingClassifier(),
                                               DecisionTreeClassifier()],
                            feature_fraction=0.8, bootstrap=True)
    clf.fit(X=train_x, y=train_y)
    for i in clf._sample_indexer:
        print("sample ratio", len(set(i))*1.0/len(i))

    pred = clf.predict_proba(X=test_x)[:, 1]
    auc_val = roc_auc_score(y_true=test_y, y_score=pred)
    print(auc_val)

    # test 4 sample fraction
    clf = BaggingClassifier(base_learner_list=[RandomForestClassifier(), GradientBoostingClassifier(),
                                               DecisionTreeClassifier()],
                            feature_fraction=0.8, bootstrap=False, sample_fraction=0.9)
    clf.fit(X=train_x, y=train_y)
    for i in clf._sample_indexer:
        print("sample ratio", len(set(i))*1.0/len(train_x))

    pred = clf.predict_proba(X=test_x)[:, 1]
    auc_val = roc_auc_score(y_true=test_y, y_score=pred)
    print(auc_val)

    # test 5 get_model_metric k_fold
    clf = BaggingClassifier(base_learner_list=[RandomForestClassifier(), GradientBoostingClassifier(),
                                               DecisionTreeClassifier()],
                            feature_fraction=0.8, bootstrap=False, sample_fraction=0.9,
                            get_model_metric=True, metric_func=roc_auc_score, metric_k_fold=5,
                            predict_strategy="weight")
    clf.fit(X=train_x, y=train_y)

    pred = clf.predict_proba(X=test_x)[:, 1]
    auc_val = roc_auc_score(y_true=test_y, y_score=pred)
    print(auc_val)

    # test 6 get_model_metric base_train_size
    clf = BaggingClassifier(base_learner_list=[RandomForestClassifier(), GradientBoostingClassifier(),
                                               DecisionTreeClassifier()],
                            feature_fraction=0.8, bootstrap=False, sample_fraction=0.9,
                            get_model_metric=True, metric_func=roc_auc_score, metric_base_train_size=0.7,
                            predict_strategy="weight")
    clf.fit(X=train_x, y=train_y)

    pred = clf.predict_proba(X=test_x)[:, 1]
    auc_val = roc_auc_score(y_true=test_y, y_score=pred)
    print(auc_val)

    # test 7 get_model_metric metric_sample_size
    clf = BaggingClassifier(base_learner_list=[RandomForestClassifier(), GradientBoostingClassifier(),
                                               DecisionTreeClassifier()],
                            feature_fraction=0.8, bootstrap=False, sample_fraction=0.9,
                            get_model_metric=True, metric_sample_size=0.8,
                            metric_func=roc_auc_score, metric_base_train_size=0.7,
                            predict_strategy="weight")
    clf.fit(X=train_x, y=train_y)

    pred = clf.predict_proba(X=test_x)[:, 1]
    auc_val = roc_auc_score(y_true=test_y, y_score=pred)
    print(auc_val)

    # test 8 get_model_metric metric_to_weight
    def metric_to_weight(metrics):
        model_weight = np.array(metrics)
        model_weight = model_weight / sum(model_weight)
        return model_weight

    clf = BaggingClassifier(base_learner_list=[RandomForestClassifier(), GradientBoostingClassifier(),
                                               DecisionTreeClassifier()],
                            feature_fraction=0.8, bootstrap=False, sample_fraction=0.9,
                            get_model_metric=True, metric_sample_size=0.8,
                            metric_func=roc_auc_score, metric_base_train_size=0.7,
                            metric_to_weight=metric_to_weight,
                            predict_strategy="weight", random_state=222)
    clf.fit(X=train_x, y=train_y)

    pred = clf.predict_proba(X=test_x)[:, 1]
    auc_val = roc_auc_score(y_true=test_y, y_score=pred)
    print(auc_val)

    # test 9 metric stratify
    def metric_to_weight(metrics):
        model_weight = np.array(metrics)
        model_weight = model_weight / sum(model_weight)
        return model_weight

    clf = BaggingClassifier(base_learner_list=[RandomForestClassifier(), GradientBoostingClassifier(),
                                               DecisionTreeClassifier()],
                            feature_fraction=0.8, bootstrap=False, sample_fraction=0.9,
                            get_model_metric=True, metric_sample_size=1,
                            metric_func=roc_auc_score, metric_base_train_size=0.7,
                            metric_to_weight=metric_to_weight,
                            predict_strategy="weight", random_state=222)
    clf.fit(X=train_x, y=train_y, metric_stratify=True, metric_stratify_col=train_y)

    pred = clf.predict_proba(X=test_x)[:, 1]
    auc_val = roc_auc_score(y_true=test_y, y_score=pred)
    print(auc_val)

    # test 10 multiprocessing
    def metric_to_weight(metrics):
        model_weight = np.array(metrics)
        model_weight = model_weight / sum(model_weight)
        return model_weight

    clf = BaggingClassifier(base_learner_list=[RandomForestClassifier(), GradientBoostingClassifier(),
                                               DecisionTreeClassifier()],
                            feature_fraction=0.8, bootstrap=False, sample_fraction=0.9,
                            get_model_metric=True, metric_sample_size=0.8,
                            metric_func=roc_auc_score, metric_base_train_size=0.7,
                            metric_to_weight=metric_to_weight,
                            predict_strategy="weight", enable_multiprocess=True,
                            n_jobs=2, random_state=222)
    clf.fit(X=train_x, y=train_y)

    pred = clf.predict_proba(X=test_x)[:, 1]
    auc_val = roc_auc_score(y_true=test_y, y_score=pred)
    print(auc_val)

    # test BaggingRegressor
    X, y = make_regression(n_samples=5000, n_features=20, random_state=224)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)

    # test1 test normal process
    clf = BaggingRegressor(base_learner_list=[RandomForestRegressor(), GradientBoostingRegressor(),
                                              DecisionTreeRegressor()])
    clf.fit(X=train_x, y=train_y)

    pred = clf.predict(X=test_x)
    r2_val = r2_score(y_true=test_y, y_pred=pred)
    print(r2_val)

    # test2 test feature fraction
    clf = BaggingRegressor(base_learner_list=[RandomForestRegressor(), GradientBoostingRegressor(),
                                              DecisionTreeRegressor()],
                           feature_fraction=0.8)
    clf.fit(X=train_x, y=train_y)
    print(clf._feature_indexer)

    pred = clf.predict(X=test_x)
    r2_val = r2_score(y_true=test_y, y_pred=pred)
    print(r2_val)

    # test3 bootstrap
    clf = BaggingRegressor(base_learner_list=[RandomForestRegressor(), GradientBoostingRegressor(),
                                              DecisionTreeRegressor()],
                           feature_fraction=0.8, bootstrap=True)
    clf.fit(X=train_x, y=train_y)
    for i in clf._sample_indexer:
        print("sample ratio", len(set(i))*1.0/len(i))

    pred = clf.predict(X=test_x)
    r2_val = r2_score(y_true=test_y, y_pred=pred)
    print(r2_val)

    # test 4 sample fraction
    clf = BaggingRegressor(base_learner_list=[RandomForestRegressor(), GradientBoostingRegressor(),
                                              DecisionTreeRegressor()],
                           feature_fraction=0.8, bootstrap=False, sample_fraction=0.9)
    clf.fit(X=train_x, y=train_y)
    for i in clf._sample_indexer:
        print("sample ratio", len(set(i))*1.0/len(train_x))

    pred = clf.predict(X=test_x)
    r2_val = r2_score(y_true=test_y, y_pred=pred)
    print(r2_val)

    # test 5 get_model_metric k_fold
    clf = BaggingRegressor(base_learner_list=[RandomForestRegressor(), GradientBoostingRegressor(),
                                              DecisionTreeRegressor()],
                           feature_fraction=0.8, bootstrap=False, sample_fraction=0.9,
                           get_model_metric=True, metric_func=r2_score, metric_k_fold=5,
                           predict_strategy="weight")
    clf.fit(X=train_x, y=train_y)

    pred = clf.predict(X=test_x)
    r2_val = r2_score(y_true=test_y, y_pred=pred)
    print(r2_val)

    # test 6 get_model_metric base_train_size
    clf = BaggingRegressor(base_learner_list=[RandomForestRegressor(), GradientBoostingRegressor(),
                                              DecisionTreeRegressor()],
                           feature_fraction=0.8, bootstrap=False, sample_fraction=0.9,
                           get_model_metric=True, metric_func=r2_score, metric_base_train_size=0.7,
                           predict_strategy="weight")
    clf.fit(X=train_x, y=train_y)

    pred = clf.predict(X=test_x)
    r2_val = r2_score(y_true=test_y, y_pred=pred)
    print(r2_val)

    # test 7 get_model_metric metric_sample_size
    clf = BaggingRegressor(base_learner_list=[RandomForestRegressor(), GradientBoostingRegressor(),
                                              DecisionTreeRegressor()],
                           feature_fraction=0.8, bootstrap=False, sample_fraction=0.9,
                           get_model_metric=True, metric_sample_size=0.8,
                           metric_func=r2_score, metric_base_train_size=0.7,
                           predict_strategy="weight")
    clf.fit(X=train_x, y=train_y)

    pred = clf.predict(X=test_x)
    r2_val = r2_score(y_true=test_y, y_pred=pred)
    print(r2_val)

    # test 8 get_model_metric metric_to_weight
    def metric_to_weight(metrics):
        model_weight = np.array(metrics)
        model_weight = model_weight / sum(model_weight)
        return model_weight

    clf = BaggingRegressor(base_learner_list=[RandomForestRegressor(), GradientBoostingRegressor(),
                                              DecisionTreeRegressor()],
                           feature_fraction=0.8, bootstrap=False, sample_fraction=0.9,
                           get_model_metric=True, metric_sample_size=0.8,
                           metric_func=r2_score, metric_base_train_size=0.7,
                           metric_to_weight=metric_to_weight,
                           predict_strategy="weight", random_state=222)
    clf.fit(X=train_x, y=train_y)

    pred = clf.predict(X=test_x)
    r2_val = r2_score(y_true=test_y, y_pred=pred)
    print(r2_val)

    # test 9 multiprocessing
    def metric_to_weight(metrics):
        model_weight = np.array(metrics)
        model_weight = model_weight / sum(model_weight)
        return model_weight

    clf = BaggingRegressor(base_learner_list=[RandomForestRegressor(), GradientBoostingRegressor(),
                                              DecisionTreeRegressor()],
                           feature_fraction=0.8, bootstrap=False, sample_fraction=0.9,
                           get_model_metric=True, metric_sample_size=0.8,
                           metric_func=r2_score, metric_base_train_size=0.7,
                           metric_to_weight=metric_to_weight,
                           predict_strategy="weight", enable_multiprocess=True,
                           n_jobs=2, random_state=222)
    clf.fit(X=train_x, y=train_y)

    pred = clf.predict(X=test_x)
    r2_val = r2_score(y_true=test_y, y_pred=pred)
    print(r2_val)
