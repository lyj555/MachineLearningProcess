# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap


def _check_feature_group(cols, feature_group, feature_meaning_dic):
    if feature_group is None:
        t = 0
        for i in cols:
            if i not in feature_meaning_dic:
                t += 1
        print(f"data columns num is {len(cols)}, feature_group set None, "
              f"so default feature group is 1, all columns. {t} features not in feature_meaning_dic,"
              f"these features will use the feature name.")
    else:
        t = 0
        feature_num = 0
        feature_set = set()
        for feature_subset in feature_group:
            feature_num += len(feature_subset)
            for feature in feature_subset:
                feature_set.add(feature)
                if feature not in cols:
                    raise ValueError(f"feature name {feature} not in data columns!")
                else:
                    if feature not in feature_meaning_dic:
                        t += 1
        assert len(feature_set) == feature_num, "feature group contain duplicated columns!"
        print(f"data columns num is {len(cols)}, feature_group num {len(feature_group)}, "
              f"contain feature num is {feature_num}.")
        if t > 0:
            print(f"in feature_group, {t} features not in feature_meaning_dic, "
                  f"these features will use the feature name.")


def _feature_to_index(cols, feature_group):
    if feature_group is None:
        return [tuple(range(len(cols)))]
    else:
        col_index_dic = dict(zip(cols, range(len(cols))))
        ret = []
        for i in feature_group:
            tmp = []
            for j in i:
                tmp.append(col_index_dic[j])
            ret.append(tuple(tmp))
    return ret


def _sort_feature_groups_by_feature_importance(feature_groups, feature_importance_list):
    if len(feature_groups) == 1:
        return feature_groups
    else:
        return sorted(feature_groups, key=lambda x: sum(feature_importance_list[list(x)]), reverse=True)


def _distribute_feature_num(feature_groups, top_feature_num):
    if len(feature_groups) > top_feature_num:
        return feature_groups[:top_feature_num], (1, )*top_feature_num
    else:
        s, y = divmod(top_feature_num, len(feature_groups))
        num_ret = [s]*len(feature_groups)
        for i in range(y):
            num_ret[i] += i
        return feature_groups, num_ret


def _select_top_feature(shap_values, feature_group, num_feature):
    feature_index_dic = dict(zip(range(len(feature_group)), feature_group))
    indices = np.argpartition(shap_values[:, feature_group], -num_feature, axis=1)[:, -num_feature:]
    indice_to_feature_func = np.vectorize(lambda x: feature_index_dic.get(x))
    return indice_to_feature_func(indices)


def _default_verbal_express(feature_value_pair, feature_meaning_dic):
    ret = []
    for feature, feature_value in feature_value_pair:
        feature_meaning = feature_meaning_dic.get(feature, feature)  # if no this key meaning, use key instead
        ret.append(f"{feature_meaning} value is {feature_value}")
    return ",".join(ret)


def _stitch_feature(array_1d, top_feature_num, index_col_dic, feature_meaning_dic, verbal_express):
    feature_index = array_1d[-top_feature_num:]
    feature_value_pair = []
    for i in feature_index:
        index = int(i)
        feature_value_pair.append((index_col_dic[index], array_1d[index]))
    if verbal_express is None:
        return _default_verbal_express(feature_value_pair, feature_meaning_dic)
    else:
        return verbal_express(feature_value_pair, feature_meaning_dic)


def lgb_explain(data, model, top_feature_num, strategy, prob_threshold=0.5, feature_group=None,
                if_sort_group_by_feature_importance=False, feature_meaning_dic={}, verbal_express=None):
    """
    use shap explain lightgbm model, for now only support binary classification
    :param data: pandas.DataFrame, data need to predict
    :param model: lgb.Booster or sklearn's lgb, lightgbm's model
    :param top_feature_num: int, >=1, the feature number use to explain model score
    :param strategy: str[max_shap, min_shap, abs_shap, threshold], use to process shap value
    :param prob_threshold: None or float, only valid when strategy set threshold, default 0.5.
    :param feature_group: None or list[tuple], use to specify the feature group,
    set to None means use all columns as one group, default None
    :param if_sort_group_by_feature_importance: bool, use feature importance to sort feature group,
    only valid when feature_group is not None
    :param feature_meaning_dic: dict, feature meaning dictionary, use to clarify the meaning of feature, default {}
    :param verbal_express: None or callable object, use to stitch the expression, default None
    :return: list[str], the length = len(data)
    """
    assert isinstance(data, pd.DataFrame), "input data must be pandas.DataFrame"
    assert isinstance(model, lgb.basic.Booster) or isinstance(model, lgb.LGBMModel), \
        "input model must be lightgbm's model"
    assert isinstance(top_feature_num, int) and top_feature_num >= 1, \
        "input top_feature_num must be int and greater or equal than 1!"
    assert strategy in ("max_shap", "min_shap", "abs_shap", "threshold"), \
        "input strategy should in ('max_shap', 'min_shap', 'abs_shap', 'threshold')!"
    if strategy == "threshold" and prob_threshold is None:
        raise ValueError("when strategy set threshold, prob_threshold can not be None!")
    assert verbal_express is None or callable(verbal_express), "input verbal_express must be None or callable object!"

    cols = data.columns
    _check_feature_group(cols, feature_group, feature_meaning_dic)
    if isinstance(model, lgb.LGBMModel):
        model = model.booster_

    objective_name = ["objective", "objective_type", "app", "application"]
    for i in objective_name:
        objective = model.params.get(i, None)
        if objective is not None:
            break

    if objective == "binary":
        feature_groups = _feature_to_index(cols, feature_group)
        if if_sort_group_by_feature_importance:
            feature_importances = model.feature_importance()
            feature_groups = _sort_feature_groups_by_feature_importance(feature_groups, feature_importances)

        feature_groups, nums = _distribute_feature_num(feature_groups, top_feature_num)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data)[1]  # len(data)*len(feat)
        if strategy == "max_shap":
            ret = None
            for index, feature_group in enumerate(feature_groups):
                indices = _select_top_feature(shap_values, feature_group, nums[index])
                ret = indices if ret is None else np.concatenate([ret, indices], axis=1)
        elif strategy == "min_shap":
            ret = None
            for index, feature_group in enumerate(feature_groups):
                indices = _select_top_feature(-shap_values, feature_group, nums[index])
                ret = indices if ret is None else np.concatenate([ret, indices], axis=1)
        elif strategy == "abs_shap":
            ret = None
            for index, feature_group in enumerate(feature_groups):
                indices = _select_top_feature(np.abs(shap_values), feature_group, nums[index])
                ret = indices if ret is None else np.concatenate([ret, indices], axis=1)
        elif strategy == "threshold":
            pos_ret = None
            for index, feature_group in enumerate(feature_groups):
                indices = _select_top_feature(shap_values, feature_group, nums[index])
                pos_ret = indices if pos_ret is None else np.concatenate([pos_ret, indices], axis=1)

            ret = None
            for index, feature_group in enumerate(feature_groups):
                indices = _select_top_feature(-shap_values, feature_group, nums[index])
                ret = indices if ret is None else np.concatenate([ret, indices], axis=1)

            pred = model.predict(data)
            pos_neg_select = np.where(pred >= prob_threshold, True, False)
            ret[pos_neg_select] = pos_ret[pos_neg_select]
        index_col_dic = dict(zip(range(len(cols)), cols))
        data_index = np.concatenate([data, ret], axis=1)

        # ret = np.apply_along_axis(lambda x: _stitch_feature(x, top_feature_num, index_col_dic,
        #                                                     feature_meaning_dic, verbal_express),
        #                           axis=1, arr=data_index)
        ret = []
        for row in data_index:
            ret.append(_stitch_feature(row, top_feature_num, index_col_dic,
                                       feature_meaning_dic, verbal_express))
        return ret
    else:
        raise ValueError("objective supported binary for now!")


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    X, y = shap.datasets.adult()

    # create a train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    d_train = lgb.Dataset(X_train, label=y_train)
    d_test = lgb.Dataset(X_test, label=y_test)

    params = {
        "max_bin": 512,
        "learning_rate": 0.05,
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 10,
        "verbose": -1,
        "min_data": 100,
        "boost_from_average": True
    }

    model = lgb.train(params, d_train, 10000, valid_sets=[d_test], early_stopping_rounds=50, verbose_eval=1000)

    # test strategy
    # for i in ["max_shap", "min_shap", "abs_shap", "threshold"]:
    #     reasons = lgb_explain(X_test, model, top_feature_num=3, strategy=i, feature_group=None)
    #     print(reasons)
    #
    # # test feature group
    feature_group = [("Age", "Sex", "Workclass"), ("Education-Num", "Marital Status", "Occupation", "Capital Loss"),
                     ("Hours per week", "Capital Gain", "Relationship")]
    # reasons = lgb_explain(X_test, model, top_feature_num=3, strategy="max_shap", feature_group=feature_group)
    # reasons = lgb_explain(X_test, model, top_feature_num=3, strategy="max_shap",
    #                       feature_group=feature_group, if_sort_group_by_feature_importance=True)
    #
    # # test feature_meaning_dic
    # feature_meaning_dic = {i: f"{i}_{i}" for i in X_test.columns}
    # reasons = lgb_explain(X_test, model, top_feature_num=3, strategy="max_shap",
    #                       feature_group=feature_group, if_sort_group_by_feature_importance=True,
    #                       feature_meaning_dic=feature_meaning_dic)
    #
    # test verbal express
    feature_meaning_dic = {i: f"{i}_{i}" for i in X_test.columns}


    def self_func(feature_value_pair, feature_meaning_dic):
        ret = []
        for feature, feature_value in feature_value_pair:
            feature_meaning = feature_meaning_dic.get(feature, feature)  # if no this key meaning, use key instead
            ret.append(f"{feature_meaning} value is {feature_value}")
        return ",".join(ret)

    reasons = lgb_explain(X_test, model, top_feature_num=3, strategy="max_shap",
                          feature_group=feature_group, if_sort_group_by_feature_importance=True,
                          feature_meaning_dic=feature_meaning_dic, verbal_express=test_verbal_express)
    print(reasons)
