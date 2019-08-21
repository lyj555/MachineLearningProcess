# -*- coding: utf-8 -*-


def cross_validation_score(train, y, k_fold, model, **kwargs):
    from sklearn.model_selection import cross_val_score

    score_list = cross_val_score(estimator=model, X=train, y=y, cv=k_fold, **kwargs)
    return sum(score_list)/len(score_list)


def valid_set_score(train_x, train_y, valid_x, valid_y, model, metric_func, **kwargs):
    clf = model.fit(train_x, train_y, **kwargs)
    valid_pred = clf.predict_proba(valid_x)[:,1]
    return metric_func(valid_y, valid_pred)
