# -*- coding: utf-8 -*-

from datetime import datetime
import warnings
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import ClassifierMixin, RegressorMixin, is_classifier, is_regressor

from .ensemble_base import BaggingBase
from ..utils.common_utils import check_X_y
from ..utils.constants import JobType, ModelStage


class BaggingClassifier(BaggingBase, ClassifierMixin):
    """Bagging for classification
    An ensemble learning, which include one stage learning, base learner train(many base learners)
    and integrate base learner's prediction to get last prediction.

    :param base_learner_list: list(sklearn Classifier estimators), base learner, default [].
    :param feature_fraction: float, (0,1], used to specify the feature fraction ration of base learner. default 1.0
    :param bootstrap: bool, if enable bootstrap sample, default False.
    :param sample_fraction: float, (0, 1], when disable bootstrap, can self-defined sample ratio, default 1.0
    :param metric_func: None or callable, used to estimate the effect of base learner, default None.
    :param get_model_metric: bool, used to specify if get base learner metric,
    if enable, the param name with string metric need to specify, default False
    :param metric_sample_size: float, (0, 1], the fraction of training data used to get model metric. default 1.0
    :param metric_k_fold: None or int or splitter object(KFold or StratifiedKFold), get metric by k fold, default None
    :param metric_base_train_size: None or int or float, get metric by train valid set, default None.
    :param metric_to_weight: str('softmax') or callable object, transform the metric into probability distribution,
    default softmax.
    :param predict_strategy: {'mean', 'weight'}, if mean, average the prediction, if weight, weighted sum the prediction
    , default mean.
    :param verbose: bool, whether or not print training messages. default True
    :param enable_multiprocess: bool, whether or not enable multiprocessing.
    :param n_jobs: int, geq 1, when enable multiprocessing, param `n_jobs` must be geq 1's integer.
    :param distribute: bool, whether or not enable spark distribute mode.
    :param spark: Spark Session, when enable distribute mode, param `spark` must be a valid Spark Session.
    :param num_partition: None or int, when enable distribute mode, the param used to specify the parallel degree
    if set None, it will determine automatically the partitions based on resources.
    :param random_state: None or int, random state used to reproduce results.
    """
    def __init__(self, base_learner_list=[], feature_fraction=1.0,
                 bootstrap=False, sample_fraction=1.0, metric_func=None, get_model_metric=False,
                 metric_sample_size=1.0, metric_k_fold=None, metric_base_train_size=None,
                 metric_to_weight="softmax", predict_strategy="mean",
                 verbose=True, enable_multiprocess=False, n_jobs=2,
                 distribute=False, spark=None, num_partition=None, random_state=None):
        super().__init__(base_learner_list=base_learner_list, metric_func=metric_func,
                         feature_fraction=feature_fraction, bootstrap=bootstrap,
                         sample_fraction=sample_fraction, get_model_metric=get_model_metric,
                         metric_sample_size=metric_sample_size, metric_k_fold=metric_k_fold,
                         metric_base_train_size=metric_base_train_size, metric_to_weight=metric_to_weight,
                         predict_strategy=predict_strategy, verbose=verbose,
                         enable_multiprocess=enable_multiprocess, n_jobs=n_jobs, distribute=distribute,
                         spark=spark, num_partition=num_partition, random_state=random_state)

    def fit(self, X, y, metric_stratify=False, metric_stratify_col=None):
        assert isinstance(self.base_learner_list, list) and len(self.base_learner_list) >= 1, \
            "param base_learner_list must be list and length geq 1!"
        for index, val in enumerate(self.base_learner_list):
            assert is_classifier(val), f"param base_learner_list index {index} is not a valid classifier!"
        assert self.metric_func is None or (self.metric_func is not None and callable(self.metric_func)), \
            "param metric_func is None or callable object!"
        assert 0 < self.feature_fraction <= 1, "param feature_fraction must in (0, 1]!"
        assert isinstance(self.bootstrap, bool), "param bootstrap must be bool type!"
        if not self.bootstrap:
            assert 0 < self.sample_fraction <= 1, "when disable bootstrap, param sample_fraction must in (0,1]!"
        assert isinstance(self.get_model_metric, bool), "param get_model_metric must be bool type!"
        if self.get_model_metric:
            assert callable(self.metric_func), \
                "when get_model_metric set True, param metric_func must be a callable object!"
            assert 0 < self.metric_sample_size <= 1, \
                "when get_model_metric set True, param metric_sample_size must in (0, 1]!"
            assert self.metric_k_fold is not None or self.metric_base_train_size is not None, \
                "when get_model_metric set True, param metric_k_fold and metric_base_train_size can not be None " \
                "at same time!"
            if self.metric_k_fold is not None:
                assert (isinstance(self.metric_k_fold, int) and self.metric_k_fold >= 2) or \
                       isinstance(self.metric_k_fold, KFold) or isinstance(self.metric_k_fold, StratifiedKFold), \
                       "when metric_k_fold is not None, " \
                       "param metric_k_fold must be int and geq 2 or KFold or StratifiedKFold object!"
            if self.metric_base_train_size is not None:
                assert (isinstance(self.metric_base_train_size, int) and self.metric_base_train_size >= 1) or \
                       (isinstance(self.metric_base_train_size, float) and (0 < self.metric_base_train_size < 1)), \
                       "when metric_k_fold is not None and metric_base_train_size is not None, " \
                       "param metric_base_train_size must be int and geq 1 or float in (0, 1)!"
            assert self.metric_to_weight in ("softmax", ) or callable(self.metric_to_weight), \
                "when metric_k_fold is not None, param metric_to_weight must be softmax or a callable object!"

            assert isinstance(metric_stratify, bool), "param metric_stratify must bool type!"
            if metric_stratify:
                assert self.metric_sample_size == 1, "when param metric_stratify set True, " \
                                                         "param metric_sample_size must be 1!"
                assert metric_stratify_col is not None, "when param metric_stratify set True, " \
                                                        "param metric_stratify_col must be a array like object!"
        assert self.predict_strategy in ("mean", "weight"), \
            "param predict_strategy must be in ('mean', 'weight')!"
        if self.predict_strategy == "weight":
            assert self.get_model_metric, "when param predict_strategy set True, param get_model_metric must be True!"

        if self.predict_strategy == "mean" and self.get_model_metric:
            warnings.warn("when param get_model_metric set True, the param predict_strategy for now is str,"
                          "this means the model metric is not used, which may consume extra time, suggest you"
                          "set get_model_metric=True or set predict_strategy callable object!")

        assert isinstance(self.distribute, bool), "param distribute must bool type!"
        if self.distribute:
            assert self.spark is not None, "when param distribute set True, param spark must be a Spark Session!"

        X, y, self._feature_info = check_X_y(X=X, y=y, stage=ModelStage.TRAIN, job=JobType.CLASSIFICATION)
        # [1]. train base learner
        print("[1]. train base learner...")
        t1 = datetime.now()
        self._base_learner_train(X=X, y=y)
        t2 = datetime.now()
        print(f"[1]. train base learner done, cost {(t2-t1).seconds} seconds.")

        if self.get_model_metric:
            print("[2]. get metric...")
            t1 = datetime.now()
            metric_sample_index = self._random_get_indexes(range(X.shape[0]), self.metric_sample_size, self.random_state)
            # print("metric_sample_size ratio", len(metric_sample_index)*1.0/len(X))
            self._get_base_learner_metric(X=X[metric_sample_index, :], y=y[metric_sample_index],
                                          job=JobType.CLASSIFICATION, metric_stratify=metric_stratify,
                                          metric_stratify_col=metric_stratify_col)
            self._model_metric_to_weight()
            t2 = datetime.now()
            print(f"[2]. get metric done, cost {(t2-t1).seconds}.")
        print("train done.")
        return self

    def predict_proba(self, X):
        X = check_X_y(X=X, y=None, stage=ModelStage.PREDICT, job=JobType.CLASSIFICATION,
                      feature_info=self._feature_info)
        base_learner_preds = self._get_base_learner_pred(X=X, job=JobType.CLASSIFICATION)
        last_pred = self._weight_predict(base_pred=base_learner_preds, weight=self._model_weight,
                                         predict_stragegy=self.predict_strategy, job=JobType.CLASSIFICATION)
        return last_pred

    def predict(self, X):
        proba = self.predict_proba(X=X)[:, 1]
        return (proba > 0.5).astype(int)


class BaggingRegressor(BaggingBase, RegressorMixin):
    """Bagging for regression
    An ensemble learning, which include one stage learning, base learner train(many base learners)
    and integrate base learner's prediction to get last prediction.

    :param base_learner_list: list(sklearn Regressor estimators), base learner, default [].
    :param feature_fraction: float, (0,1], used to specify the feature fraction ration of base learner. default 1.0
    :param bootstrap: bool, if enable bootstrap sample, default False.
    :param sample_fraction: float, (0, 1], when disable bootstrap, can self-defined sample ratio, default 1.0
    :param metric_func: None or callable, used to estimate the effect of base learner, default None.
    :param get_model_metric: bool, used to specify if get base learner metric,
    if enable, the param name with string metric need to specify, default False
    :param metric_sample_size: float, (0, 1], the fraction of training data used to get model metric. default 1.0
    :param metric_k_fold: None or int or splitter object(KFold or StratifiedKFold), get metric by k fold, default None
    :param metric_base_train_size: None or int or float, get metric by train valid set, default None.
    :param metric_to_weight: str('softmax') or callable object, transform the metric into probability distribution,
    default softmax.
    :param predict_strategy: {'mean', 'weight'}, if mean, average the prediction, if weight, weighted sum the prediction
    , default mean.
    :param verbose: bool, whether or not print training messages. default True
    :param enable_multiprocess: bool, whether or not enable multiprocessing.
    :param n_jobs: int, geq 1, when enable multiprocessing, param `n_jobs` must be geq 1's integer.
    :param distribute: bool, whether or not enable spark distribute mode.
    :param spark: Spark Session, when enable distribute mode, param `spark` must be a valid Spark Session.
    :param num_partition: None or int, when enable distribute mode, the param used to specify the parallel degree
    if set None, it will determine automatically the partitions based on resources.
    :param random_state: None or int, random state used to reproduce results.
    """
    def __init__(self, base_learner_list=[], feature_fraction=1.0,
                 bootstrap=False, sample_fraction=1.0, metric_func=None, get_model_metric=False,
                 metric_sample_size=1.0, metric_k_fold=None, metric_base_train_size=None,
                 metric_to_weight="softmax", predict_strategy="mean",
                 verbose=True, enable_multiprocess=False, n_jobs=2,
                 distribute=False, spark=None, num_partition=None, random_state=None):
        super().__init__(base_learner_list=base_learner_list, metric_func=metric_func,
                         feature_fraction=feature_fraction, bootstrap=bootstrap,
                         sample_fraction=sample_fraction, get_model_metric=get_model_metric,
                         metric_sample_size=metric_sample_size, metric_k_fold=metric_k_fold,
                         metric_base_train_size=metric_base_train_size, metric_to_weight=metric_to_weight,
                         predict_strategy=predict_strategy, verbose=verbose,
                         enable_multiprocess=enable_multiprocess, n_jobs=n_jobs, distribute=distribute,
                         spark=spark, num_partition=num_partition, random_state=random_state)

    def fit(self, X, y, metric_stratify=False, metric_stratify_col=None):
        assert isinstance(self.base_learner_list, list) and len(self.base_learner_list) >= 1, \
            "param base_learner_list must be list and length geq 1!"
        for index, val in enumerate(self.base_learner_list):
            assert is_regressor(val), f"param base_learner_list index {index} is not a valid classifier!"
        assert self.metric_func is None or (self.metric_func is not None and callable(self.metric_func)), \
            "param metric_func is None or callable object!"
        assert 0 < self.feature_fraction <= 1, "param feature_fraction must in (0, 1]!"
        assert isinstance(self.bootstrap, bool), "param bootstrap must be bool type!"
        if not self.bootstrap:
            assert 0 < self.sample_fraction <= 1, "when disable bootstrap, param sample_fraction must in (0,1]!"
        assert isinstance(self.get_model_metric, bool), "param get_model_metric must be bool type!"
        if self.get_model_metric:
            assert callable(self.metric_func), \
                "when get_model_metric set True, param metric_func must be a callable object!"
            assert 0 < self.metric_sample_size <= 1, \
                "when get_model_metric set True, param metric_sample_size must in (0, 1]!"
            assert self.metric_k_fold is not None or self.metric_base_train_size is not None, \
                "when get_model_metric set True, param metric_k_fold and metric_base_train_size can not be None " \
                "at same time!"
            if self.metric_k_fold is not None:
                assert (isinstance(self.metric_k_fold, int) and self.metric_k_fold >= 2) or \
                       isinstance(self.metric_k_fold, KFold) or isinstance(self.metric_k_fold, StratifiedKFold), \
                       "when metric_k_fold is not None, " \
                       "param metric_k_fold must be int and geq 2 or KFold or StratifiedKFold object!"
            if self.metric_base_train_size is not None:
                assert (isinstance(self.metric_base_train_size, int) and self.metric_base_train_size >= 1) or \
                       (isinstance(self.metric_base_train_size, float) and (0 < self.metric_base_train_size < 1)), \
                       "when metric_k_fold is not None and metric_base_train_size is not None, " \
                       "param metric_base_train_size must be int and geq 1 or float in (0, 1)!"
            assert self.metric_to_weight in ("softmax", ) or callable(self.metric_to_weight), \
                "when metric_k_fold is not None, param metric_to_weight must be softmax or a callable object!"

            assert isinstance(metric_stratify, bool), "param metric_stratify must bool type!"
            if metric_stratify:
                assert self.metric_sample_size == 1, "when param metric_stratify set True, " \
                                                         "param metric_sample_size must be 1!"
                assert metric_stratify_col is not None, "when param metric_stratify set True, " \
                                                        "param metric_stratify_col must be a array like object!"

        assert self.predict_strategy in ("mean", "weight"), \
            "param predict_strategy must be in ('mean', 'weight')!"
        if self.predict_strategy == "weight":
            assert self.get_model_metric, "when param predict_strategy set True, param get_model_metric must be True!"

        if self.predict_strategy == "mean" and self.get_model_metric:
            warnings.warn("when param get_model_metric set True, the param predict_strategy for now is str,"
                          "this means the model metric is not used, which may consume extra time, suggest you"
                          "set get_model_metric=True or set predict_strategy callable object!")

        assert isinstance(self.distribute, bool), "param distribute must bool type!"
        if self.distribute:
            assert self.spark is not None, "when param distribute set True, param spark must be a Spark Session!"

        X, y, self._feature_info = check_X_y(X=X, y=y, stage=ModelStage.TRAIN, job=JobType.REGRESSION)
        # [1]. train base learner
        print("[1]. train base learner...")
        t1 = datetime.now()
        self._base_learner_train(X=X, y=y)
        t2 = datetime.now()
        print(f"[1]. train base learner done, cost {(t2-t1).seconds} seconds.")

        if self.get_model_metric:
            print("[2]. get metric...")
            t1 = datetime.now()
            metric_sample_index = self._random_get_indexes(range(X.shape[0]), self.metric_sample_size, self.random_state)
            # print("metric_sample_size ratio", len(metric_sample_index)*1.0/len(X))
            self._get_base_learner_metric(X=X[metric_sample_index, :], y=y[metric_sample_index],
                                          job=JobType.REGRESSION, metric_stratify=metric_stratify,
                                          metric_stratify_col=metric_stratify_col)
            self._model_metric_to_weight()
            t2 = datetime.now()
            print(f"[2]. get metric done, cost {(t2-t1).seconds}.")
        print("train done.")
        return self

    def predict(self, X):
        X = check_X_y(X=X, y=None, stage=ModelStage.PREDICT, job=JobType.REGRESSION,
                      feature_info=self._feature_info)
        base_learner_preds = self._get_base_learner_pred(X=X, job=JobType.REGRESSION)
        last_pred = self._weight_predict(base_pred=base_learner_preds, weight=self._model_weight,
                                         predict_stragegy=self.predict_strategy, job=JobType.REGRESSION)
        return last_pred

