# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import train_test_split

from bayes_opt import BayesianOptimization

from utils.model_utils import cross_validation_score, valid_set_score


def bayes_search(train_x, train_y, model, param_space, n_iter, k_fold=None, create_valid=False, valid_ratio=None,
                 valid_x=None, valid_y=None, metric_func=None, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    param_bounds, param_types = {}, {}
    # initialize param interval and type
    for i in param_space:
        param_bounds[i] = param_space[i]["interval"]
        param_types[i] = param_space[i]["type"]

    if create_valid and valid_ratio is not None and (0 < valid_ratio < 1):
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=valid_ratio,
                                                              random_state=random_state)

    def optimize_func(**kwargs):
        for i in kwargs:
            kwargs[i] = param_types[i](kwargs[i])

        if k_fold is not None:
            ret = cross_validation_score(train_x, train_y, k_fold, model, model_param=kwargs)
        else:
            ret = valid_set_score(train_x, train_y, valid_x, valid_y, model=model,
                                  metric_func=metric_func, model_param=kwargs)
        return ret

    optimizer = BayesianOptimization(
        f=optimize_func,
        pbounds=param_bounds,
        random_state=random_state,
        verbose=2
    )
    optimizer.maximize(n_iter=n_iter)

    opt_ret = optimizer.max
    for i in param_types:
        opt_ret["params"][i] = param_types[i](opt_ret["params"][i])
    best_result, best_params = opt_ret["target"], opt_ret["params"]

    return best_result, best_params


# if __name__ == "__main__":
#     from sklearn.datasets import make_classification
#     from sklearn.ensemble import RandomForestClassifier
#
#
#     def _create_data():
#         """Synthetic binary classification dataset."""
#         data, targets = make_classification(
#             n_samples=1000,
#             n_features=45,
#             n_informative=12,
#             n_redundant=7,
#             random_state=134985745,
#         )
#         return data, targets
#
#     data, targets = _create_data()
#     clf = RandomForestClassifier()
#
#     param_space = {"max_features": {"interval": (0.1, 0.9), "type": float},
#                    "n_estimators": {"interval": (10, 250), "type": int},
#                    "min_samples_split": {"interval": (2, 25), "type": int}
#                    }
#     best_result, best_params = bayes_search(data, targets, model=clf, param_space=param_space, n_iter=10,
#                                             k_fold=3, random_state=666)
#     print(f"best_result is {best_result}, best_param is {best_params}")
