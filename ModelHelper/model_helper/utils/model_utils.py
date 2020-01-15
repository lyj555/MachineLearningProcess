# -*- coding: utf-8 -*-

import copy
import warnings
from datetime import datetime

import numpy as np
from sklearn.base import is_classifier, is_regressor


def cross_validation_score(train, y, k_fold, model, random_state=None, model_param=None, metric_func=None,
                           return_time=False, **kwargs):
    model = copy.deepcopy(model)
    t1 = datetime.now()
    from sklearn.model_selection import cross_val_score

    if random_state is not None:
        np.random.seed(random_state)

    if model_param is not None:
        model = model.set_params(**model_param)

    if "scoring" not in kwargs:
        if metric_func is not None:
            if is_classifier(model):
                score_func = lambda clf, X, y: metric_func(y_true=y, y_score=clf.predict_proba(X)[:, 1])
            elif is_regressor(model):
                score_func = lambda clf, X, y: metric_func(y_true=y, y_pred=clf.predict(X))
            else:
                warnings.warn("input model is not classifier neither nor regressor, may encounter unexpected error!")
                score_func = lambda clf, X, y: metric_func(y, clf.predict(X))
        else:
            score_func = None
        score_list = cross_val_score(estimator=model, X=train, y=y, cv=k_fold, scoring=score_func, **kwargs)
    else:
        score_list = cross_val_score(estimator=model, X=train, y=y, cv=k_fold, **kwargs)
    t2 = datetime.now()
    if return_time:
        return sum(score_list) / len(score_list), (t2 - t1).seconds
    else:
        return sum(score_list) / len(score_list)


def valid_set_score(train_x, train_y, valid_x, valid_y, model, metric_func=None, model_param=None,
                    model_fit_param=None, return_time=False, return_model=False, set_eval_set=False):
    model = copy.deepcopy(model)
    t1 = datetime.now()
    if model_param is not None:
        model = model.set_params(**model_param)

    if model_fit_param is None:
        clf = model.fit(train_x, train_y)
    else:
        if set_eval_set:
            clf = model.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], **model_fit_param)
        else:
            clf = model.fit(train_x, train_y, **model_fit_param)
    if is_classifier(clf):
        valid_pred = clf.predict_proba(valid_x)[:, 1]
    elif is_regressor(clf):
        valid_pred = clf.predict(valid_x)
    else:
        warnings.warn("input model is not classifier neither nor regressor, may encounter unexpected error!")
        valid_pred = clf.predict(valid_x)
    t2 = datetime.now()

    if return_time and return_model:
        return metric_func(valid_y, valid_pred), (t2 - t1).seconds, clf
    elif return_time and not return_model:
        return metric_func(valid_y, valid_pred), (t2 - t1).seconds, None
    elif not return_time and return_model:
        return metric_func(valid_y, valid_pred), None, clf
    else:
        return metric_func(valid_y, valid_pred), None, None
