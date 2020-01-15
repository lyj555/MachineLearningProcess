# -*- coding: utf-8 -*-

from datetime import datetime
from sklearn.base import ClassifierMixin, RegressorMixin, is_classifier, is_regressor

from .ensemble_base import BlendingBase
from ..utils.common_utils import check_X_y
from ..utils.constants import JobType, ModelStage


class BlendingClassifier(BlendingBase, ClassifierMixin):
    """Blending for classification
        An ensemble learning, which include two stage learning, base learner train(many base learners)
        and meta learner train(only one meta learner)

        Parameters
        ----------
        :param base_train_size: int([1, )) or float((0, 1)), if int, means the number of training data,
        if float, means the fraction of training data.
        :param base_learner_list: list(sklearn Classifier estimators), base learner, default [].
        :param meta_learner: sklearn Classifier estimator, meta learner, default None.
        :param select_base_learner: None or callable, used to choose base learner from `base_learner_list`, default None.
        :param metric_func: None or callable, used to estimate the effect of base learner, default None.
        :param feature_fraction: float, (0,1], used to specify the feature fraction ration of base learner.
        :param verbose: bool, whether or not print training messages.
        :param enable_multiprocess: bool, whether or not enable multiprocessing.
        :param n_jobs: int, geq 1, when enable multiprocessing, param `n_jobs` must be geq 1's integer.
        :param distribute: bool, whether or not enable spark distribute mode.
        :param spark: Spark Session, when enable distribute mode, param `spark` must be a valid Spark Session.
        :param num_partition: None or int, when enable distribute mode, the param used to specify the parallel degree,
        if set None, it will determine automatically the partitions based on resources.
        :param random_state: None or int, random state used to reproduce results.

        Attributes
        ----------
        base_learner_num_: int
            The number of input base learner
        feature_indexer_: list[np.array(int)]
            The feature indexer of every base learner model
        sample_indexer_: list[np.array(int)]
            The sample splitting indexer of every base learner model
        select_model_index_: list[int]
            The base learner model index last used
        """
    def __init__(self, base_train_size=0.6, base_learner_list=[], meta_learner=None,
                 select_base_learner=None, metric_func=None, feature_fraction=1.0,
                 verbose=True, enable_multiprocess=False, n_jobs=2,
                 distribute=False, spark=None, num_partition=None, random_state=None):
        super().__init__(base_train_size=base_train_size, base_learner_list=base_learner_list,
                         meta_learner=meta_learner, select_base_learner=select_base_learner,
                         metric_func=metric_func, feature_fraction=feature_fraction, verbose=verbose,
                         enable_multiprocess=enable_multiprocess, n_jobs=n_jobs, distribute=distribute,
                         spark=spark, num_partition=num_partition, random_state=random_state)

    def fit(self, X, y, stratify=False, stratify_col=None):
        """
        method to train BlendingClassifier
        :param X: pandas DataFrame or numpy.ndarray, training data
        :param y: pd.Series or numpy.ndarray, one-dimension data
        :param stratify: bool, whether or not stratify data, the param `stratify_col` must be specified.
        :param stratify_col: pd.Series or numpy.ndarray, one-dimension data,
        :return: self
        """
        assert (isinstance(self.base_train_size, int) and self.base_train_size >= 1) or \
               (isinstance(self.base_train_size, float) and (0 < self.base_train_size < 1)), \
               "param base_train_size must be int and geq 1 or float in (0, 1)!"
        assert isinstance(self.base_learner_list, list) and len(self.base_learner_list) >= 1, \
            "param base_learner_list must be list and length geq 1!"
        for index, val in enumerate(self.base_learner_list):
            assert is_classifier(val), f"param base_learner_list index {index} is not a valid classifier!"
        assert is_classifier(self.meta_learner), "f param meta_learner is not a valid classifier!"
        assert self.select_base_learner is None or \
               (self.select_base_learner is not None and callable(self.select_base_learner)), \
            "param select_base_learner is None or callable object!"
        assert self.metric_func is None or (self.metric_func is not None and callable(self.metric_func)), \
            "param metric_func is None or callable object!"
        if self.select_base_learner is not None:
            assert callable(self.metric_func), \
                "when param selector is not None, param metric_func must be a callable object!"
        assert (isinstance(self.feature_fraction, float) and (0 < self.feature_fraction <= 1)), \
            "param feature_fraction must be (0, 1]!"
        assert isinstance(self.distribute, bool), "param distribute must bool type!"
        if self.distribute:
            assert self.spark is not None, "when param distribute set True, param spark must be a Spark Session!"

        assert isinstance(stratify, bool), "param stratify must bool type!"
        if stratify:
            assert stratify_col is not None, "when param stratify set True, " \
                                                  "param stratify_col must be a array like object!"

        X, y, self._feature_info = check_X_y(X=X, y=y, stage=ModelStage.TRAIN, job=JobType.CLASSIFICATION)
        # [1]. train base learner
        print("[1]. train base learner...")
        t1 = datetime.now()
        self._base_learner_train(X=X, y=y, stratify=stratify, stratify_col=stratify_col)
        t2 = datetime.now()
        print(f"[1]. train base learner done, cost {(t2-t1).seconds} seconds.")

        # [2]. predict data sets use base trained base learner
        print("[2]. get base learner prediction...")
        t1 = datetime.now()
        meta_feature, model_metric = self._base_learner_train_get_pred_and_metric(X=X, y=y,
                                                                                  job=JobType.CLASSIFICATION)
        if self.verbose:
            print(f"meta learner used sample num is {meta_feature.shape[0]}, "
                  f"fraction is {meta_feature.shape[0]/X.shape[0]}")
        t2 = datetime.now()
        print(f"[2]. get base learner prediction done, cost {(t2-t1).seconds} seconds.")

        print("[3]. train meta learner...")
        t1 = datetime.now()
        self._select_base_learner(model_metric=model_metric)
        self._meta_learner_train(meta_feature=meta_feature, y=y)
        t2 = datetime.now()
        print(f"[3]. train meta learner done, cost {(t2-t1).seconds} seconds.")
        return self

    def predict_proba(self, X):
        X = check_X_y(X=X, y=None, stage=ModelStage.PREDICT, job=JobType.CLASSIFICATION,
                      feature_info=self._feature_info)
        base_learner_feature = self._get_base_learner_pred(X=X, job=JobType.CLASSIFICATION)
        last_pred = self.meta_learner.predict_proba(base_learner_feature)
        return last_pred

    def predict(self, X):
        proba = self.predict_proba(X=X)[:, 1]
        return (proba > 0.5).astype(int)


class BlendingRegressor(BlendingBase, RegressorMixin):
    """Blending for regression
        An ensemble learning, which include two stage learning, base learner train(many base learners)
        and meta learner train(only one meta learner)

        Parameters
        ----------
        :param base_train_size: int([1, )) or float((0, 1)), if int, means the number of training data,
        if float, means the fraction of training data.
        :param base_learner_list: list(sklearn Regressor estimators), base learner, default [].
        :param meta_learner: sklearn Regressor estimator, meta learner, default None.
        :param select_base_learner: None or callable, used to choose base learner from `base_learner_list`, default None.
        :param metric_func: None or callable, used to estimate the effect of base learner, default None.
        :param feature_fraction: float, (0,1], used to specify the feature fraction ration of base learner.
        :param verbose: bool, whether or not print training messages.
        :param enable_multiprocess: bool, whether or not enable multiprocessing.
        :param n_jobs: int, geq 1, when enable multiprocessing, param `n_jobs` must be geq 1's integer.
        :param distribute: bool, whether or not enable spark distribute mode.
        :param spark: Spark Session, when enable distribute mode, param `spark` must be a valid Spark Session.
        :param num_partition: None or int, when enable distribute mode, the param used to specify the parallel degree,
        if set None, it will determine automatically the partitions based on resources.
        :param random_state: None or int, random state used to reproduce results.

        Attributes
        ----------
        base_learner_num_: int
            The number of input base learner
        feature_indexer_: list[np.array(int)]
            The feature indexer of every base learner model
        sample_indexer_: list[np.array(int)]
            The sample splitting indexer of every base learner model
        select_model_index_: list[int]
            The base learner model index last used
        """
    def __init__(self, base_train_size=0.6, base_learner_list=[], meta_learner=None,
                 select_base_learner=None, metric_func=None, feature_fraction=1.0,
                 verbose=True, enable_multiprocess=False, n_jobs=2,
                 distribute=False, spark=None, num_partition=None, random_state=None):
        super().__init__(base_train_size=base_train_size, base_learner_list=base_learner_list,
                         meta_learner=meta_learner, select_base_learner=select_base_learner,
                         metric_func=metric_func, feature_fraction=feature_fraction, verbose=verbose,
                         enable_multiprocess=enable_multiprocess, n_jobs=n_jobs, distribute=distribute,
                         spark=spark, num_partition=num_partition, random_state=random_state)

    def fit(self, X, y, stratify=False, stratify_col=None):
        assert (isinstance(self.base_train_size, int) and self.base_train_size >= 1) or \
               (isinstance(self.base_train_size, float) and (0 < self.base_train_size < 1)), \
            "param base_train_size must be int and geq 1 or float in (0, 1)!"
        assert isinstance(self.base_learner_list, list) and len(self.base_learner_list) >= 1, \
            "param base_learner_list must be list and length geq 1!"
        for index, val in enumerate(self.base_learner_list):
            assert is_regressor(val), f"param base_learner_list index {index} is not a valid regressor!"
        assert is_regressor(self.meta_learner), "f param meta_learner is not a valid regressor!"
        assert self.select_base_learner is None or \
               (self.select_base_learner is not None and callable(self.select_base_learner)), \
            "param select_base_learner is None or callable object!"
        assert self.metric_func is None or (self.metric_func is not None and callable(self.metric_func)), \
            "param metric_func is None or callable object!"
        if self.select_base_learner is not None:
            assert callable(self.metric_func), \
                "when param selector is not None, param metric_func must be a callable object!"
        assert (isinstance(self.feature_fraction, float) and (0 < self.feature_fraction <= 1)), \
            "param feature_fraction must be (0, 1]!"
        assert isinstance(self.distribute, bool), "param distribute must bool type!"
        if self.distribute:
            assert self.spark is not None, "when param distribute set True, param spark must be a Spark Session!"

        assert isinstance(stratify, bool), "param stratify must bool type!"
        if stratify:
            assert stratify_col is not None, "when param stratify set True, " \
                                                  "param stratify_col must be a array like object!"

        X, y, self._feature_info = check_X_y(X=X, y=y, stage=ModelStage.TRAIN, job=JobType.REGRESSION)
        # [1]. train base learner
        print("[1]. train base learner...")
        t1 = datetime.now()
        self._base_learner_train(X=X, y=y, stratify=stratify, stratify_col=stratify_col)
        t2 = datetime.now()
        print(f"[1]. train base learner done, cost {(t2-t1).seconds} seconds.")

        # [2]. predict data sets use base trained base learner
        print("[2]. get base learner prediction...")
        t1 = datetime.now()
        meta_feature, model_metric = self._base_learner_train_get_pred_and_metric(X=X, y=y,
                                                                                  job=JobType.REGRESSION)
        if self.verbose:
            print(f"meta learner used sample num is {meta_feature.shape[0]}, "
                  f"fraction is {meta_feature.shape[0]/X.shape[0]}")
        t2 = datetime.now()
        print(f"[2]. get base learner prediction done, cost {(t2-t1).seconds} seconds.")

        print("[3]. train meta learner...")
        t1 = datetime.now()
        self._select_base_learner(model_metric=model_metric)
        self._meta_learner_train(meta_feature=meta_feature, y=y)
        t2 = datetime.now()
        print(f"[3]. train meta learner done, cost {(t2-t1).seconds} seconds.")
        return self

    def predict(self, X):
        X = check_X_y(X=X, y=None, stage=ModelStage.PREDICT, job=JobType.REGRESSION,
                      feature_info=self._feature_info)
        base_learner_feature = self._get_base_learner_pred(X=X, job=JobType.REGRESSION)
        last_pred = self.meta_learner.predict(base_learner_feature)
        return last_pred
