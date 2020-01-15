# -*- coding: utf-8 -*-

# from hyperopt import fmin, tpe, hp
# best = fmin(fn=lambda x: x ** 2,
#             space=hp.uniform('x', -10, 10),
#             algo=tpe.suggest,
#             max_evals=1000)
# print(best)

# import pickle
# import time
# from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
#
#
# def objective(x):
#     return {
#         'loss': x ** 2,
#         'status': STATUS_OK,
#         # -- store other results like this
#         'eval_time': time.time(),
#         'other_stuff': {'x': x, 'value': [0, 1, 2]},
#         # -- attachments are handled differently
#         'attachments':
#             {'time_module': pickle.dumps(time.time)}
#         }
#
#
# trials = Trials()
# best = fmin(objective,
#             space=hp.uniform('x', -10, 10),
#             algo=tpe.suggest,
#             max_evals=10,
#             trials=trials)
#
# print(best)
#
# for i in trials.trials:
#     print(i["result"]["other_stuff"]["x"])


# from hyperopt import hp
# import hyperopt.pyll.stochastic
#
# space = hp.choice('a',
#             [
#                 ('case 1', 1 + hp.lognormal('c1', 0, 1)),
#                 ('case 2', hp.uniform('c2', -10, 10))
#             ])
#
# print(hyperopt.pyll.stochastic.sample(space))


import numpy as np
from sklearn.model_selection import train_test_split

from ..utils.model_utils import cross_validation_score, valid_set_score


def hyperopt_search(train_x, train_y, model, param_space, n_iter, k_fold=None, create_valid=False, valid_ratio=None,
                    valid_x=None, valid_y=None, metric_func=None, random_state=None):
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
    if k_fold is None and not callable(metric_func):
        raise ValueError("if k_fold set None, param metric_func must be callable object!")
    if random_state is not None:
        np.random.seed(random_state)

    if create_valid and valid_ratio is not None and (0 < valid_ratio < 1):
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=valid_ratio,
                                                              random_state=random_state)

    def optimize_func(param_space):
        if k_fold is not None:
            ret = cross_validation_score(train_x, train_y, k_fold, model, model_param=param_space,
                                         random_state=random_state)
        else:
            ret = valid_set_score(train_x, train_y, valid_x, valid_y, model=model,
                                  metric_func=metric_func, model_param=param_space)[0]
        return {"loss": -ret, "status": STATUS_OK, "stuff": {"param": param_space, "effect": ret}}

    trials = Trials()
    best_params = fmin(optimize_func,
                       space=param_space,
                       algo=tpe.suggest,
                       max_evals=n_iter,
                       trials=trials,
                       rstate=np.random.RandomState(random_state))
    # best_result, best_params = opt_ret["target"], opt_ret["params"]

    return trials, best_params


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


    def _create_data():
        """Synthetic binary classification dataset."""
        data, targets = make_classification(
            n_samples=1000,
            n_features=45,
            n_informative=12,
            n_redundant=7,
            random_state=134985745,
        )
        return data, targets

    data, targets = _create_data()
    clf = RandomForestClassifier()

    param_space = {"max_features": hp.uniform("max_features", 0.1, 0.9),
                   "n_estimators": hp.choice("n_estimators", range(10, 100)),
                   "min_samples_split": hp.choice("min_samples_split", [2, 10, 100])
                   }
    trials, best_params = hyperopt_search(data, targets, model=clf, param_space=param_space, n_iter=10,
                                          k_fold=3, random_state=666)
    for i in trials:
        print(i["result"]["stuff"])
    print(f"best_param is {best_params}")


# 目前 hyperopt 的优化算法所识别的随机表达式是：
#
# hp.choice(label, options)
#
# 返回其中一个选项，它应该是一个列表或元组。options元素本身可以是[嵌套]随机表达式。在这种情况下，仅出现在某些选项中的随机选择(stochastic choices)将成为条件参数。
# hp.randint(label, upper)
#
# 返回范围:[0，upper]中的随机整数。当更远的整数值相比较时,这种分布的语义是意味着邻整数值之间的损失函数没有更多的相关性。例如，这是描述随机种子的适当分布。如果损失函数可能更多的与相邻整数值相关联，那么你或许应该用“量化”连续分布的一个，比如 quniform ， qloguniform ， qnormal 或 qlognormal 。
# hp.uniform(label, low, high)
#
# 返回位于[low,hight]之间的均匀分布的值。
# 在优化时，这个变量被限制为一个双边区间。
# hp.quniform(label, low, high, q)
#
# 返回一个值，如 round（uniform（low，high）/ q）* q
# 适用于目标仍然有点“光滑”的离散值，但是在它上下存在边界(双边区间)。
# hp.loguniform(label, low, high)
#
# 返回根据 exp（uniform（low，high）） 绘制的值，以便返回值的对数是均匀分布的。 优化时，该变量被限制在[exp（low），exp（high）]区间内。
# hp.qloguniform(label, low, high, q)
#
# 返回类似 round（exp（uniform（low，high））/ q）* q 的值
# 适用于一个离散变量，其目标是“平滑”，并随着值的大小变得更平滑，但是在它上下存在边界(双边区间)。
# hp.normal(label, mu, sigma)
#
# 返回正态分布的实数值，其平均值为 mu ，标准偏差为 σ。优化时，这是一个无约束(unconstrained)的变量。
# hp.qnormal(label, mu, sigma, q)
#
# 返回像 round（正常（mu，sigma）/ q）* q 的值
# 适用于离散值，可能需要在 mu 附近的取值，但从基本上上是无约束(unbounded)的。
# hp.lognormal(label, mu, sigma)(对数正态分布)
#
# 返回根据 exp（normal（mu，sigma）） 绘制的值，以便返回值的对数正态分布。优化时，这个变量被限制为正值。
# hp.qlognormal(label, mu, sigma, q)
#
# 返回类似 round（exp（normal（mu，sigma））/ q）* q 的值
# 适用于一个离散变量，其目标是“平滑”，并随着值的大小变得更平滑，变量的大小是从一个边界开始的(单边区间)。
