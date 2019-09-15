# -*- coding: utf-8 -*-

from datetime import datetime


def cross_validation_score(train, y, k_fold, model, model_param=None, metric_func=None, return_model_train_time=False, **kwargs):
    t1 = datetime.now()
    from sklearn.model_selection import cross_val_score

    if model_param is not None:
        model = model.set_params(**model_param)
    if metric_func is not None:
        score_func = lambda clf, X, y: metric_func(y_true=y, y_pred=clf.predict_proba(X)[:, 1])
    else:
        score_func = None
    score_list = cross_val_score(estimator=model, X=train, y=y, cv=k_fold, scoring=score_func, **kwargs)
    t2 = datetime.now()
    if return_model_train_time:
        return sum(score_list) / len(score_list), (t2 - t1).seconds
    else:
        return sum(score_list) / len(score_list)


def valid_set_score(train_x, train_y, valid_x, valid_y, model, metric_func=None, model_param=None,
                    return_model_train_time=False):
    t1 = datetime.now()
    if model_param is not None:
        model = model.set_params(**model_param)
    clf = model.fit(train_x, train_y)
    valid_pred = clf.predict_proba(valid_x)[:, 1]
    t2 = datetime.now()
    if return_model_train_time:
        return metric_func(valid_y, valid_pred), (t2 - t1).seconds
    else:
        return metric_func(valid_y, valid_pred)
