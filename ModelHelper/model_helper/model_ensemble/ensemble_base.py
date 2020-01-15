# -*- coding: utf-8 -*-

from multiprocessing import Pool
import numpy as np
from copy import copy
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.base import BaseEstimator

from ..utils.spark_utils import dynamic_confirm_partition_num, uniform_partition
from ..utils.common_utils import get_n_random_list
from ..utils.constants import JobType


class StackingBase(BaseEstimator):
    def __init__(self, k_fold=5, base_learner_list=[], meta_learner=None,
                 select_base_learner=None, metric_func=None, feature_fraction=1.0,
                 verbose=True, enable_multiprocess=False, n_jobs=2,
                 distribute=False, spark=None, num_partition=None, random_state=None):
        self.k_fold = k_fold
        self.base_learner_list = base_learner_list
        self.meta_learner = meta_learner
        self.select_base_learner = select_base_learner
        self.metric_func = metric_func
        self.feature_fraction = feature_fraction
        self.verbose = verbose
        self.enable_multiprocess = enable_multiprocess
        self.n_jobs = n_jobs
        self.distribute = distribute
        self.spark = spark
        self.num_partition = num_partition
        self.random_state = random_state
        self._feature_info = None

    def __construct_k_fold_base_leaner_info(self, splitter, sample_num, feature_num, stratify_col):
        self.k_fold_, self.base_learner_num_ = splitter.get_n_splits(), len(self.base_learner_list)
        if isinstance(self.feature_fraction, list):
            self.feature_indexer_ = self.feature_fraction
        else:
            self.feature_indexer_ = get_n_random_list(alternative_list=range(feature_num), fraction=self.feature_fraction,
                                                      n=self.base_learner_num_, random_state=self.random_state)
        self.sample_indexer_ = list(splitter.split(X=np.zeros(sample_num), y=stratify_col))
        k_fold_base_learner_info = \
            [(model_index, fold_index) for model_index in range(self.base_learner_num_)
             for fold_index in range(self.k_fold_)]
        return k_fold_base_learner_info

    def __local_loop_k_fold_base_learner(self, X, y, fold_base_info):
        ret = []
        if self.enable_multiprocess:
            pool = Pool(processes=self.n_jobs)
            for index, val in enumerate(fold_base_info):
                model_index, fold_index = val
                if self.verbose:
                    print(f"{index+1}/{len(fold_base_info)}(model_index:{model_index}, fold_index:{fold_index}) starts")
                sample_index, feature_index = self.sample_indexer_[fold_index][0], self.feature_indexer_[model_index]
                result = pool.apply_async(func=self.train_model_by_data, args=(X[sample_index, :][:, feature_index],
                                                                               y[sample_index],
                                                                               model_index, fold_index,
                                                                               self.base_learner_list[model_index]))
                ret.append(result)
            pool.close()
            pool.join()
            ret = [i.get() for i in ret]
        else:
            for index, val in enumerate(fold_base_info):
                model_index, fold_index = val
                if self.verbose:
                    print(f"{index+1}/{len(fold_base_info)}(model_index:{model_index}, fold_index:{fold_index}) starts")
                sample_index, feature_index = self.sample_indexer_[fold_index][0], self.feature_indexer_[model_index]
                result = self.train_model_by_data(X=X[sample_index, :][:, feature_index], y=y[sample_index],
                                                  model_index=model_index, fold_index=fold_index,
                                                  model=self.base_learner_list[model_index])
                ret.append(result)
        return ret

    def __construct_model_index_fold_index_model_dic(self, ret):
        self._model_dic = {}
        for model_index, fold_index, model in ret:
            if model_index not in self._model_dic:
                self._model_dic[model_index] = {}
            self._model_dic[model_index][fold_index] = model

    def _base_learner_train_get_pred_and_metric(self, X, y, job):
        meta_feature, model_metric = [], []
        for model_index in range(self.base_learner_num_):
            tmp_feature, tmp_metric = np.array([]), []
            for fold_index in range(self.k_fold_):
                model = self._model_dic[model_index][fold_index]
                sample_index, feature_index = self.sample_indexer_[fold_index][1], self.feature_indexer_[model_index]

                if job == JobType.CLASSIFICATION:
                    tmp_pred = model.predict_proba(X[sample_index, :][:, feature_index])[:, 1]
                elif job == JobType.REGRESSION:
                    tmp_pred = model.predict(X[sample_index, :][:, feature_index])
                else:
                    raise ValueError("param job should in ('classification', 'regression')!")

                if self.metric_func is not None:
                    metric = self.metric_func(y[sample_index], tmp_pred)
                    tmp_metric.append(metric)
                tmp_feature = np.append(tmp_feature, tmp_pred)

            meta_feature.append(tmp_feature)
            if self.metric_func is not None:
                model_metric.append(tmp_metric)
        if self.verbose and self.metric_func is not None:
            print(f"average {self.k_fold_} fold metric of every model "
                  f"is {list(map(lambda x: sum(x)/len(x), model_metric))}")
        return np.array(meta_feature).T, model_metric

    def _select_base_learner(self, model_metric):
        if self.select_base_learner is None:
            self.select_model_index_ = list(range(self.base_learner_num_))
        else:
            self.select_model_index_ = np.array(self.select_base_learner(model_metric))

    def _meta_learner_train(self, meta_feature, y):
        train_data = meta_feature[:, self.select_model_index_]
        if self.verbose:
            print(f"last used model index is {self.select_model_index_}.")

        # reconstruct the y's position
        y_index = np.array([], dtype=int)
        for fold_index in range(self.k_fold_):
            _, test_index = self.sample_indexer_[fold_index]
            y_index = np.append(y_index, test_index)
        y = y[y_index]

        self.meta_learner.fit(train_data, y)

    def _base_learner_train(self, X, y, splitter, stratify_col):
        sample_num, feature_num = X.shape
        # splitter = self.__get_splitter()  # get splitter for split data into k_fold
        # k_fold * base_learner_num's list
        fold_base_info = self.__construct_k_fold_base_leaner_info(splitter=splitter, sample_num=sample_num,
                                                                  feature_num=feature_num, stratify_col=stratify_col)
        if self.verbose:
            print(f"altogether train {self.base_learner_num_}(base learner model number) × "
                  f"{self.k_fold_}(fold num) = {len(fold_base_info)} models.")
        if not self.distribute:
            fold_base_results = self.__local_loop_k_fold_base_learner(X=X, y=y, fold_base_info=fold_base_info)
        else:
            fold_base_results = self.__distribute_loop_k_fold_base_learner(X=X, y=y, fold_base_info=fold_base_info,
                                                                           spark=self.spark,
                                                                           base_learner_list=self.base_learner_list,
                                                                           sample_indexer=self.sample_indexer_,
                                                                           feature_indexer=self.feature_indexer_,
                                                                           num_partition=self.num_partition)

        # convert model result to dic, add self
        self.__construct_model_index_fold_index_model_dic(fold_base_results)

    def _get_base_learner_pred(self, X, job):
        try:
            base_pred = []
            for model_index in self.select_model_index_:
                cur_pred = None
                for fold_index in range(self.k_fold_):
                    model = self._model_dic[model_index][fold_index]  # get trained model

                    if job == JobType.CLASSIFICATION:
                        tmp_pred = model.predict_proba(X[:, self.feature_indexer_[model_index]])[:, 1]
                    elif job == JobType.REGRESSION:
                        tmp_pred = model.predict(X[:, self.feature_indexer_[model_index]])
                    else:
                        raise ValueError("param job wrong!")

                    if cur_pred is None:
                        cur_pred = tmp_pred
                    else:
                        cur_pred += tmp_pred
                base_pred.append(cur_pred / self.k_fold_)
        except AttributeError:
            raise ValueError("model not fitted, first call `fit` method!")
        return np.array(base_pred).T

    @staticmethod
    def _get_splitter(k_fold, random_state, stratify):
        if isinstance(k_fold, int) and k_fold >= 2:
            if stratify:
                splitter = StratifiedKFold(n_splits=k_fold, random_state=random_state, shuffle=True)
            else:
                splitter = KFold(n_splits=k_fold, random_state=random_state, shuffle=True)
        else:
            splitter = k_fold
        return splitter

    @staticmethod
    def _partition_loop(X, y, indexes, base_learner_list, sample_indexer, feature_indexer):
        X, y = X.value, y.value
        ret = []
        for _, val in indexes:
            model_index, fold_index = val
            sample_index, feature_index = sample_indexer[fold_index][0], feature_indexer[model_index]
            result = StackingBase.train_model_by_data(X=X[sample_index, :][:, feature_index], y=y[sample_index],
                                                      model_index=model_index, fold_index=fold_index,
                                                      model=base_learner_list[model_index])
            ret.append(result)
        return ret

    @staticmethod
    def train_model_by_data(X, y, model_index, fold_index, model):
        model = copy(model)
        model.fit(X, y)
        return model_index, fold_index, model

    @staticmethod
    def __distribute_loop_k_fold_base_learner(spark, X, y, fold_base_info, base_learner_list, sample_indexer,
                                              feature_indexer, num_partition):
        if num_partition is None:
            num_partition = dynamic_confirm_partition_num(spark.sparkContext)
        else:
            num_partition = num_partition

        X, y = spark.sparkContext.broadcast(X), spark.sparkContext.broadcast(y)
        s = uniform_partition(spark=spark, content_list=fold_base_info,
                              num_partition=num_partition)
        partition_result = s.mapPartitions(lambda x: StackingBase._partition_loop(X=X, y=y, indexes=x,
                                                                                  base_learner_list=base_learner_list,
                                                                                  sample_indexer=sample_indexer,
                                                                                  feature_indexer=feature_indexer)).\
            collect()
        return partition_result


class BlendingBase(BaseEstimator):
    def __init__(self, base_train_size=0.6, base_learner_list=[], meta_learner=None,
                 select_base_learner=None, metric_func=None, feature_fraction=1.0,
                 verbose=True, enable_multiprocess=False, n_jobs=2,
                 distribute=False, spark=None, num_partition=None, random_state=None):
        self.base_train_size = base_train_size
        self.base_learner_list = base_learner_list
        self.meta_learner = meta_learner
        self.select_base_learner = select_base_learner
        self.metric_func = metric_func
        self.feature_fraction = feature_fraction
        self.verbose = verbose
        self.enable_multiprocess = enable_multiprocess
        self.n_jobs = n_jobs
        self.distribute = distribute
        self.spark = spark
        self.num_partition = num_partition
        self.random_state = random_state
        self._feature_info = None

    def __construct_base_leaner_info(self, sample_num, feature_num, stratify, stratify_col):
        self.base_learner_num_ = len(self.base_learner_list)

        if isinstance(self.feature_fraction, list):
            self.feature_indexer_ = self.feature_fraction
        else:
            self.feature_indexer_ = get_n_random_list(alternative_list=range(feature_num), fraction=self.feature_fraction,
                                                      n=self.base_learner_num_, random_state=self.random_state)
        self.sample_indexer_ = self._get_sample_split_index(base_train_size=self.base_train_size,
                                                            stratify=stratify, stratify_col=stratify_col,
                                                            sample_num=sample_num, random_state=self.random_state)
        base_learner_info = list(range(self.base_learner_num_))
        return base_learner_info

    def __local_loop_base_learner(self, X, y, base_info):
        ret = []
        if self.enable_multiprocess:
            pool = Pool(processes=self.n_jobs)
            for index, model_index in enumerate(base_info):
                if self.verbose:
                    print(f"{index+1}/{len(base_info)}(model_index:{model_index}) starts")
                sample_index, feature_index = self.sample_indexer_[0], self.feature_indexer_[model_index]
                result = pool.apply_async(func=self.train_model_by_data, args=(X[sample_index, :][:, feature_index],
                                                                               y[sample_index], model_index,
                                                                               self.base_learner_list[model_index]))
                ret.append(result)
            pool.close()
            pool.join()
            ret = [i.get() for i in ret]
        else:
            for index, model_index in enumerate(base_info):
                if self.verbose:
                    print(f"{index+1}/{len(base_info)}(model_index:{model_index}) starts")
                sample_index, feature_index = self.sample_indexer_[0], self.feature_indexer_[model_index]
                result = self.train_model_by_data(X=X[sample_index, :][:, feature_index], y=y[sample_index],
                                                  model_index=model_index, model=self.base_learner_list[model_index])
                ret.append(result)
        return ret

    def __construct_model_index_dic(self, ret):
        self._model_dic = {}
        for model_index, model in ret:
            self._model_dic[model_index] = model

    def _base_learner_train_get_pred_and_metric(self, X, y, job):
        meta_feature, model_metric = [], []
        for model_index in range(self.base_learner_num_):
            model = self._model_dic[model_index]
            sample_index, feature_index = self.sample_indexer_[1], self.feature_indexer_[model_index]

            if job == JobType.CLASSIFICATION:
                tmp_pred = model.predict_proba(X[sample_index, :][:, feature_index])[:, 1]
            elif job == JobType.REGRESSION:
                tmp_pred = model.predict(X[sample_index, :][:, feature_index])
            else:
                raise ValueError("param job should in ('classification', 'regression')!")

            if self.metric_func is not None:
                metric = self.metric_func(y[sample_index], tmp_pred)
                model_metric.append(metric)

            meta_feature.append(tmp_pred)
        if self.verbose and self.metric_func is not None:
            print(f"model's test set metric is {model_metric}.")
        return np.array(meta_feature).T, model_metric

    def _select_base_learner(self, model_metric):
        if self.select_base_learner is None:
            self.select_model_index_ = list(range(self.base_learner_num_))
        else:
            self.select_model_index_ = np.array(self.select_base_learner(model_metric))

    def _meta_learner_train(self, meta_feature, y):
        train_data = meta_feature[:, self.select_model_index_]
        if self.verbose:
            print(f"last used model index is {self.select_model_index_}.")

        # reconstruct the y's position
        y_index = self.sample_indexer_[1]
        y = y[y_index]

        self.meta_learner.fit(train_data, y)

    def _base_learner_train(self, X, y, stratify, stratify_col):
        sample_num, feature_num = X.shape
        base_info = self.__construct_base_leaner_info(sample_num=sample_num, feature_num=feature_num,
                                                      stratify=stratify, stratify_col=stratify_col)
        if self.verbose:
            print(f"altogether train {self.base_learner_num_} models.")
            print(f"base learner used sample num is {len(self.sample_indexer_[0])}, "
                  f"fraction is {len(self.sample_indexer_[0])/X.shape[0]}")
        if not self.distribute:
            fold_base_results = self.__local_loop_base_learner(X=X, y=y, base_info=base_info)
        else:
            fold_base_results = self.__distribute_loop_base_learner(X=X, y=y, base_info=base_info,
                                                                    spark=self.spark,
                                                                    base_learner_list=self.base_learner_list,
                                                                    sample_indexer=self.sample_indexer_,
                                                                    feature_indexer=self.feature_indexer_,
                                                                    num_partition=self.num_partition)

        # convert model result to dic, add self
        self.__construct_model_index_dic(fold_base_results)

    def _get_base_learner_pred(self, X, job):
        try:
            base_pred = []
            for model_index in self.select_model_index_:
                model = self._model_dic[model_index]  # get trained model

                if job == JobType.CLASSIFICATION:
                    tmp_pred = model.predict_proba(X[:, self.feature_indexer_[model_index]])[:, 1]
                elif job == JobType.REGRESSION:
                    tmp_pred = model.predict(X[:, self.feature_indexer_[model_index]])
                else:
                    raise ValueError("param job wrong!")

                base_pred.append(tmp_pred)
        except AttributeError:
            raise ValueError("model not fitted, first call `fit` method!")
        return np.array(base_pred).T

    @staticmethod
    def _partition_loop(X, y, indexes, base_learner_list, sample_indexer, feature_indexer):
        X, y = X.value, y.value
        ret = []
        for _, model_index in indexes:
            sample_index, feature_index = sample_indexer[0], feature_indexer[model_index]
            result = BlendingBase.train_model_by_data(X=X[sample_index, :][:, feature_index], y=y[sample_index],
                                                      model_index=model_index, model=base_learner_list[model_index])
            ret.append(result)
        return ret

    @staticmethod
    def _get_sample_split_index(base_train_size, stratify, stratify_col, sample_num, random_state):
        if not stratify:
            sample_indexer = train_test_split(range(sample_num), train_size=base_train_size, random_state=random_state)
        else:
            sample_indexer = train_test_split(range(sample_num), train_size=base_train_size,
                                              stratify=stratify_col, random_state=random_state)
        return sample_indexer

    @staticmethod
    def train_model_by_data(X, y, model_index, model):
        model = copy(model)
        model.fit(X, y)
        return model_index, model

    @staticmethod
    def __distribute_loop_base_learner(spark, X, y, base_info, base_learner_list, sample_indexer,
                                       feature_indexer, num_partition):
        if num_partition is None:
            num_partition = dynamic_confirm_partition_num(spark.sparkContext)
        else:
            num_partition = num_partition

        X, y = spark.sparkContext.broadcast(X), spark.sparkContext.broadcast(y)
        s = uniform_partition(spark=spark, content_list=base_info,
                              num_partition=num_partition)
        partition_result = s.mapPartitions(lambda x: BlendingBase._partition_loop(X=X, y=y, indexes=x,
                                                                                  base_learner_list=base_learner_list,
                                                                                  sample_indexer=sample_indexer,
                                                                                  feature_indexer=feature_indexer)).\
            collect()
        return partition_result


class _StackingGetMetric(StackingBase):
    def __init__(self, k_fold=5, base_learner_list=[], metric_func=None, feature_fraction=1.0,
                 enable_multiprocess=False, n_jobs=2,
                 distribute=False, spark=None, num_partition=None, random_state=None):
        super().__init__(k_fold=k_fold, base_learner_list=base_learner_list,
                         meta_learner=None, select_base_learner=None,
                         metric_func=metric_func, feature_fraction=feature_fraction, verbose=False,
                         enable_multiprocess=enable_multiprocess, n_jobs=n_jobs, distribute=distribute,
                         spark=spark, num_partition=num_partition, random_state=random_state)

    def fit(self, X, y, job, stratify=False, stratify_col=None):
        splitter = super()._get_splitter(k_fold=self.k_fold, random_state=self.random_state, stratify=stratify)
        self._base_learner_train(X=X, y=y, splitter=splitter, stratify_col=stratify_col)

        _, model_metric = self._base_learner_train_get_pred_and_metric(X=X, y=y,
                                                                       job=job)
        return list(map(lambda x: sum(x)/len(x), model_metric))


class _BlendingGetMetric(BlendingBase):
    def __init__(self, base_train_size=0.6, base_learner_list=[],
                 metric_func=None, feature_fraction=1.0, enable_multiprocess=False, n_jobs=2,
                 distribute=False, spark=None, num_partition=None, random_state=None):
        super().__init__(base_train_size=base_train_size, base_learner_list=base_learner_list,
                         meta_learner=None, select_base_learner=None,
                         metric_func=metric_func, feature_fraction=feature_fraction, verbose=False,
                         enable_multiprocess=enable_multiprocess, n_jobs=n_jobs, distribute=distribute,
                         spark=spark, num_partition=num_partition, random_state=random_state)

    def fit(self, X, y, job, stratify=False, stratify_col=None):
        self._base_learner_train(X=X, y=y, stratify=stratify, stratify_col=stratify_col)
        _, model_metric = self._base_learner_train_get_pred_and_metric(X=X, y=y, job=job)
        return model_metric


class BaggingBase(BaseEstimator):
    def __init__(self, base_learner_list=[], metric_func=None, feature_fraction=1.0,
                 bootstrap=False, sample_fraction=1.0, get_model_metric=False,
                 metric_sample_size=1.0, metric_k_fold=None, metric_base_train_size=None,
                 metric_to_weight="softmax", predict_strategy="mean", verbose=True,
                 enable_multiprocess=False, n_jobs=2, distribute=False, spark=None,
                 num_partition=None, random_state=None):
        self.base_learner_list = base_learner_list
        self.metric_func = metric_func
        self.feature_fraction = feature_fraction
        self.bootstrap = bootstrap
        self.sample_fraction = sample_fraction
        self.get_model_metric = get_model_metric
        self.metric_sample_size = metric_sample_size
        self.metric_k_fold = metric_k_fold
        self.metric_base_train_size = metric_base_train_size
        self.metric_to_weight = metric_to_weight
        self.predict_strategy = predict_strategy
        self.verbose = verbose
        self.enable_multiprocess = enable_multiprocess
        self.n_jobs = n_jobs
        self.distribute = distribute
        self.spark = spark
        self.num_partition = num_partition
        self.random_state = random_state
        self._model_metric = None
        self._model_weight = None
        self._feature_info = None

    def __construct_base_leaner_info(self, sample_num, feature_num):
        assert isinstance(self.feature_fraction, float) and (0 < self.feature_fraction <= 1), \
            "param feature_fraction must be (0, 1]!"
        if self.bootstrap:
            assert isinstance(self.sample_fraction, float) and (0 < self.sample_fraction <= 1), \
                "param sample_fraction must be (0, 1]!"

        self._base_learner_num = len(self.base_learner_list)
        self._feature_indexer = get_n_random_list(alternative_list=range(feature_num), fraction=self.feature_fraction,
                                                  n=self._base_learner_num, random_state=self.random_state)
        self._sample_indexer = get_n_random_list(alternative_list=range(sample_num), fraction=self.sample_fraction,
                                                 n=self._base_learner_num, random_state=self.random_state,
                                                 bootstrap=self.bootstrap)
        base_learner_info = [model_index for model_index in range(self._base_learner_num)]
        return base_learner_info

    def __local_loop_base_learner(self, X, y, fold_base_info):
        ret = []
        if self.enable_multiprocess:
            pool = Pool(processes=self.n_jobs)
            for index, model_index in enumerate(fold_base_info):
                if self.verbose:
                    print(f"{index+1}/{len(fold_base_info)}(model_index:{model_index}) starts")
                sample_index, feature_index = self._sample_indexer[model_index], self._feature_indexer[model_index]
                result = pool.apply_async(func=self.train_model_by_data, args=(X[sample_index, :][:, feature_index],
                                                                               y[sample_index], model_index,
                                                                               self.base_learner_list[model_index]))
                ret.append(result)
            pool.close()
            pool.join()
            ret = [i.get() for i in ret]
        else:
            for index, model_index in enumerate(fold_base_info):
                if self.verbose:
                    print(f"{index+1}/{len(fold_base_info)}(model_index:{model_index}) starts")
                sample_index, feature_index = self._sample_indexer[model_index], self._feature_indexer[model_index]
                result = self.train_model_by_data(X=X[sample_index, :][:, feature_index], y=y[sample_index],
                                                  model_index=model_index, model=self.base_learner_list[model_index])
                ret.append(result)
        return ret

    def __construct_model_index_dic(self, ret):
        self._model_dic = {}
        for model_index, model in ret:
            self._model_dic[model_index] = model

    def _get_base_learner_metric(self, X, y, job, metric_stratify, metric_stratify_col):
        if self.metric_k_fold is not None:
            metric_instance = _StackingGetMetric(k_fold=self.metric_k_fold, base_learner_list=self.base_learner_list,
                                                 metric_func=self.metric_func, feature_fraction=self._feature_indexer,
                                                 enable_multiprocess=self.enable_multiprocess, n_jobs=self.n_jobs,
                                                 distribute=self.distribute, spark=self.spark,
                                                 num_partition=self.num_partition, random_state=self.random_state)
        else:
            metric_instance = _BlendingGetMetric(base_train_size=self.metric_base_train_size,
                                                 base_learner_list=self.base_learner_list, metric_func=self.metric_func,
                                                 feature_fraction=self._feature_indexer,
                                                 enable_multiprocess=self.enable_multiprocess, n_jobs=self.n_jobs,
                                                 distribute=self.distribute, spark=self.spark,
                                                 num_partition=self.num_partition, random_state=self.random_state)

        self._model_metric = metric_instance.fit(X, y, job=job,
                                                 stratify=metric_stratify, stratify_col=metric_stratify_col)

    def _model_metric_to_weight(self):
        if self.metric_to_weight == "softmax":
            self._model_weight = 1 / (1 + np.exp(-np.array(self._model_metric)))
            self._model_weight = self._model_weight / sum(self._model_weight)
        else:
            self._model_weight = self.metric_to_weight(self._model_metric)

    def _base_learner_train(self, X, y):
        sample_num, feature_num = X.shape
        # k_fold * base_learner_num's list
        fold_base_info = self.__construct_base_leaner_info(feature_num=feature_num, sample_num=sample_num)
        if self.verbose:
            print(f"altogether train {self._base_learner_num} models.")
        if not self.distribute:
            fold_base_results = self.__local_loop_base_learner(X=X, y=y, fold_base_info=fold_base_info)
        else:
            fold_base_results = self.__distribute_loop_base_learner(spark=self.spark, X=X, y=y,
                                                                    base_info=fold_base_info,
                                                                    base_learner_list=self.base_learner_list,
                                                                    sample_indexer=self._sample_indexer,
                                                                    feature_indexer=self._feature_indexer,
                                                                    num_partition=self.num_partition)
        # convert model result to dic, add self
        self.__construct_model_index_dic(fold_base_results)

    def _get_base_learner_pred(self, X, job):
        try:
            base_pred = []
            for model_index in self._model_dic:
                model = self._model_dic[model_index]  # get trained model

                if job == JobType.CLASSIFICATION:
                    tmp_pred = model.predict_proba(X[:, self._feature_indexer[model_index]])[:, 1]
                elif job == JobType.REGRESSION:
                    tmp_pred = model.predict(X[:, self._feature_indexer[model_index]])
                else:
                    raise ValueError("param job wrong!")

                base_pred.append(tmp_pred)
        except AttributeError:
            raise ValueError("model not fitted, first call `fit` method!")
        return np.array(base_pred).T

    @staticmethod
    def _weight_predict(base_pred, weight, predict_stragegy, job):
        if predict_stragegy == "mean":
            pred = base_pred.mean(axis=1)
        elif predict_stragegy == "weight":
            pred = (base_pred*np.array(weight)).sum(axis=1)
        else:
            raise ValueError("predict_stragegy not in ('mean', 'weight')!")

        if job == JobType.CLASSIFICATION:
            return np.array([1-pred, pred]).T
        else:
            return pred

    @staticmethod
    def _partition_loop(X, y, indexes, base_learner_list, sample_indexer, feature_indexer):
        X, y = X.value, y.value
        ret = []
        for _, model_index in indexes:
            sample_index, feature_index = sample_indexer[0], feature_indexer[model_index]
            result = BlendingBase.train_model_by_data(X=X[sample_index, :][:, feature_index], y=y[sample_index],
                                                      model_index=model_index, model=base_learner_list[model_index])
            ret.append(result)
        return ret

    @staticmethod
    def train_model_by_data(X, y, model_index, model):
        model = copy(model)
        model.fit(X, y)
        return model_index, model

    @staticmethod
    def _random_get_indexes(alternative_list, fraction, random_state):
        return get_n_random_list(alternative_list, n=1, fraction=fraction, random_state=random_state)[0]

    @staticmethod
    def __distribute_loop_base_learner(spark, X, y, base_info, base_learner_list, sample_indexer,
                                       feature_indexer, num_partition):
        if num_partition is None:
            num_partition = dynamic_confirm_partition_num(spark.sparkContext)
        else:
            num_partition = num_partition

        X, y = spark.sparkContext.broadcast(X), spark.sparkContext.broadcast(y)
        s = uniform_partition(spark=spark, content_list=base_info,
                              num_partition=num_partition)
        partition_result = s.mapPartitions(lambda x: BaggingBase._partition_loop(X=X, y=y, indexes=x,
                                                                                 base_learner_list=base_learner_list,
                                                                                 sample_indexer=sample_indexer,
                                                                                 feature_indexer=feature_indexer)).\
            collect()
        return partition_result

