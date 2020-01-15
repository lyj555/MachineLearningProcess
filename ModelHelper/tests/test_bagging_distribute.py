# -*- coding: utf-8 -*-

import os
import zipfile
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import click
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,\
    RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from model_helper.model_ensemble import BaggingClassifier, BaggingRegressor

warnings.filterwarnings(action="ignore", category=FutureWarning)


def unzip_job():
    print(os.listdir("./"))
    for i in os.listdir("./"):
        if i.endswith(".zip") and not i.endswith("pyspark.zip") and not i.startswith("py4j"):
            job_zip = i
            print("find the zip file name", job_zip)
            break

    with zipfile.ZipFile(job_zip, "r") as zip_ref:
        zip_ref.extractall("./")


def process_file(spark, local_file_path):
    """
    该部分为自己的处理函数
    :param local_file_path: str, 数据逗号分割，如果是拉取分区表的数据，数据最后一列为partition_dt
    :return: pandas.DataFrame, 字段顺序和输出表顺序一致且类型一致，字段名字可以任意
    """

    print("TEST BaggingClassifier...")
    X, y = make_classification(n_samples=5000, n_features=20, n_classes=2, random_state=234)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)

    # test1 test normal process
    print("test1 test normal process")
    clf = BaggingClassifier(base_learner_list=[RandomForestClassifier(), GradientBoostingClassifier(),
                                               DecisionTreeClassifier()], distribute=True, spark=spark)
    clf.fit(X=train_x, y=train_y)

    pred = clf.predict_proba(X=test_x)[:, 1]
    auc_val = roc_auc_score(y_true=test_y, y_score=pred)
    print(auc_val)
    print("test1 test normal process done.")

    # test2 test feature fraction
    print("test2 test feature fraction")
    clf = BaggingClassifier(base_learner_list=[RandomForestClassifier(), GradientBoostingClassifier(),
                                               DecisionTreeClassifier()],
                            feature_fraction=0.8, distribute=True, spark=spark)
    clf.fit(X=train_x, y=train_y)
    print(clf._feature_indexer)

    pred = clf.predict_proba(X=test_x)[:, 1]
    auc_val = roc_auc_score(y_true=test_y, y_score=pred)
    print(auc_val)
    print("test2 test feature fraction done.")

    # test3 bootstrap
    print("test3 bootstrap")
    clf = BaggingClassifier(base_learner_list=[RandomForestClassifier(), GradientBoostingClassifier(),
                                               DecisionTreeClassifier()],
                            feature_fraction=0.8, bootstrap=True, distribute=True, spark=spark)
    clf.fit(X=train_x, y=train_y)
    for i in clf._sample_indexer:
        print("sample ratio", len(set(i))*1.0/len(i))

    pred = clf.predict_proba(X=test_x)[:, 1]
    auc_val = roc_auc_score(y_true=test_y, y_score=pred)
    print(auc_val)
    print("test3 bootstrap done.")

    # test 4 sample fraction
    print("test 4 sample fraction")
    clf = BaggingClassifier(base_learner_list=[RandomForestClassifier(), GradientBoostingClassifier(),
                                               DecisionTreeClassifier()],
                            feature_fraction=0.8, bootstrap=False, sample_fraction=0.9, distribute=True, spark=spark)
    clf.fit(X=train_x, y=train_y)
    for i in clf._sample_indexer:
        print("sample ratio", len(set(i))*1.0/len(train_x))

    pred = clf.predict_proba(X=test_x)[:, 1]
    auc_val = roc_auc_score(y_true=test_y, y_score=pred)
    print(auc_val)
    print("test 4 sample fraction done.")

    # test 5 get_model_metric k_fold
    print("test 5 get_model_metric k_fold")
    clf = BaggingClassifier(base_learner_list=[RandomForestClassifier(), GradientBoostingClassifier(),
                                               DecisionTreeClassifier()],
                            feature_fraction=0.8, bootstrap=False, sample_fraction=0.9,
                            get_model_metric=True, metric_func=roc_auc_score, metric_k_fold=5,
                            predict_strategy="weight", distribute=True, spark=spark)
    clf.fit(X=train_x, y=train_y)

    pred = clf.predict_proba(X=test_x)[:, 1]
    auc_val = roc_auc_score(y_true=test_y, y_score=pred)
    print(auc_val)
    print("test 5 get_model_metric k_fold done.")

    # test 6 get_model_metric base_train_size
    print("test 6 get_model_metric base_train_size...")
    clf = BaggingClassifier(base_learner_list=[RandomForestClassifier(), GradientBoostingClassifier(),
                                               DecisionTreeClassifier()],
                            feature_fraction=0.8, bootstrap=False, sample_fraction=0.9,
                            get_model_metric=True, metric_func=roc_auc_score, metric_base_train_size=0.7,
                            predict_strategy="weight", distribute=True, spark=spark)
    clf.fit(X=train_x, y=train_y)

    pred = clf.predict_proba(X=test_x)[:, 1]
    auc_val = roc_auc_score(y_true=test_y, y_score=pred)
    print(auc_val)
    print("test 6 get_model_metric base_train_size done.")

    # test 7 get_model_metric metric_sample_size
    print("test 7 get_model_metric metric_sample_size...")
    clf = BaggingClassifier(base_learner_list=[RandomForestClassifier(), GradientBoostingClassifier(),
                                               DecisionTreeClassifier()],
                            feature_fraction=0.8, bootstrap=False, sample_fraction=0.9,
                            get_model_metric=True, metric_sample_size=0.8,
                            metric_func=roc_auc_score, metric_base_train_size=0.7,
                            predict_strategy="weight", distribute=True, spark=spark)
    clf.fit(X=train_x, y=train_y)

    pred = clf.predict_proba(X=test_x)[:, 1]
    auc_val = roc_auc_score(y_true=test_y, y_score=pred)
    print(auc_val)
    print("test 7 get_model_metric metric_sample_size done.")

    # test 8 get_model_metric metric_to_weight
    print("test 8 get_model_metric metric_to_weight...")
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
                            predict_strategy="weight", random_state=222, distribute=True, spark=spark)
    clf.fit(X=train_x, y=train_y)

    pred = clf.predict_proba(X=test_x)[:, 1]
    auc_val = roc_auc_score(y_true=test_y, y_score=pred)
    print(auc_val)
    print("test 8 get_model_metric metric_to_weight done.")

    # test 9 metric stratify
    print("test 9 metric stratify...")
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
                            predict_strategy="weight", random_state=222, distribute=True, spark=spark)
    clf.fit(X=train_x, y=train_y, metric_stratify=True, metric_stratify_col=train_y)

    pred = clf.predict_proba(X=test_x)[:, 1]
    auc_val = roc_auc_score(y_true=test_y, y_score=pred)
    print(auc_val)
    print("test 9 metric stratify done.")
    print("TEST BaggingClassifier ALL DONE.")


    # test BaggingRegressor
    print("TEST BaggingRegressor...")
    X, y = make_regression(n_samples=5000, n_features=20, random_state=224)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)

    # test1 test normal process
    print("test1 test normal process")
    clf = BaggingRegressor(base_learner_list=[RandomForestRegressor(), GradientBoostingRegressor(),
                                              DecisionTreeRegressor()], distribute=True, spark=spark)
    clf.fit(X=train_x, y=train_y)

    pred = clf.predict(X=test_x)
    r2_val = r2_score(y_true=test_y, y_pred=pred)
    print(r2_val)
    print("test1 test normal process done.")

    # test2 test feature fraction
    print("test2 test feature fraction...")
    clf = BaggingRegressor(base_learner_list=[RandomForestRegressor(), GradientBoostingRegressor(),
                                              DecisionTreeRegressor()],
                           feature_fraction=0.8, distribute=True, spark=spark)
    clf.fit(X=train_x, y=train_y)
    print(clf._feature_indexer)

    pred = clf.predict(X=test_x)
    r2_val = r2_score(y_true=test_y, y_pred=pred)
    print(r2_val)
    print("test2 test feature fraction done.")

    # test3 bootstrap
    print("test3 bootstrap...")
    clf = BaggingRegressor(base_learner_list=[RandomForestRegressor(), GradientBoostingRegressor(),
                                              DecisionTreeRegressor()],
                           feature_fraction=0.8, bootstrap=True, distribute=True, spark=spark)
    clf.fit(X=train_x, y=train_y)
    for i in clf._sample_indexer:
        print("sample ratio", len(set(i))*1.0/len(i))

    pred = clf.predict(X=test_x)
    r2_val = r2_score(y_true=test_y, y_pred=pred)
    print(r2_val)
    print("test3 bootstrap done.")

    # test 4 sample fraction
    print("test 4 sample fraction...")
    clf = BaggingRegressor(base_learner_list=[RandomForestRegressor(), GradientBoostingRegressor(),
                                              DecisionTreeRegressor()],
                           feature_fraction=0.8, bootstrap=False, sample_fraction=0.9, distribute=True, spark=spark)
    clf.fit(X=train_x, y=train_y)
    for i in clf._sample_indexer:
        print("sample ratio", len(set(i))*1.0/len(train_x))

    pred = clf.predict(X=test_x)
    r2_val = r2_score(y_true=test_y, y_pred=pred)
    print(r2_val)
    print("test 4 sample fraction done.")

    # test 5 get_model_metric k_fold
    print("test 5 get_model_metric k_fold...")
    clf = BaggingRegressor(base_learner_list=[RandomForestRegressor(), GradientBoostingRegressor(),
                                              DecisionTreeRegressor()],
                           feature_fraction=0.8, bootstrap=False, sample_fraction=0.9,
                           get_model_metric=True, metric_func=r2_score, metric_k_fold=5,
                           predict_strategy="weight", distribute=True, spark=spark)
    clf.fit(X=train_x, y=train_y)

    pred = clf.predict(X=test_x)
    r2_val = r2_score(y_true=test_y, y_pred=pred)
    print(r2_val)
    print("test 5 get_model_metric k_fold done.")

    # test 6 get_model_metric base_train_size
    print("test 6 get_model_metric base_train_size...")
    clf = BaggingRegressor(base_learner_list=[RandomForestRegressor(), GradientBoostingRegressor(),
                                              DecisionTreeRegressor()],
                           feature_fraction=0.8, bootstrap=False, sample_fraction=0.9,
                           get_model_metric=True, metric_func=r2_score, metric_base_train_size=0.7,
                           predict_strategy="weight", distribute=True, spark=spark)
    clf.fit(X=train_x, y=train_y)

    pred = clf.predict(X=test_x)
    r2_val = r2_score(y_true=test_y, y_pred=pred)
    print(r2_val)
    print("test 6 get_model_metric base_train_size done.")

    # test 7 get_model_metric metric_sample_size
    print("test 7 get_model_metric metric_sample_size...")
    clf = BaggingRegressor(base_learner_list=[RandomForestRegressor(), GradientBoostingRegressor(),
                                              DecisionTreeRegressor()],
                           feature_fraction=0.8, bootstrap=False, sample_fraction=0.9,
                           get_model_metric=True, metric_sample_size=0.8,
                           metric_func=r2_score, metric_base_train_size=0.7,
                           predict_strategy="weight", distribute=True, spark=spark)
    clf.fit(X=train_x, y=train_y)

    pred = clf.predict(X=test_x)
    r2_val = r2_score(y_true=test_y, y_pred=pred)
    print(r2_val)
    print("test 7 get_model_metric metric_sample_size done.")

    # test 8 get_model_metric metric_to_weight
    print("test 8 get_model_metric metric_to_weight...")
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
                           predict_strategy="weight", random_state=222, distribute=True, spark=spark)
    clf.fit(X=train_x, y=train_y)

    pred = clf.predict(X=test_x)
    r2_val = r2_score(y_true=test_y, y_pred=pred)
    print(r2_val)
    print("test 8 get_model_metric metric_to_weight done.")
    print("TEST BaggingRegressor ALL DONE.")


def job_main(spark, partition_dt, test_local, need_download):
    """
    任务的流程分为三部分,首先拉取Hive数据至本地,然后处理本地数据,之后将结果数据输出至Hive表
    :param spark: spark session
    :param partition_dt: str,格式%Y%m%d
    :param test_local: 取值0或1(default 0),0表示在DP上执行，1表示本地运行
    :param need_download: 取值0或1(default 1),0表示不需要从hdfs拉取数据，1需要从hdfs拉取数据
    :return: None
    """
    print("process file...")
    t1 = datetime.now()
    process_file(spark, local_file_path=None)
    t2 = datetime.now()
    print(f"process file succ, cost {(t2-t1).seconds} seconds.")


@click.command()
@click.option("--partition_dt", type=str, help="input date")
@click.option("--test_local", type=int, default=0, help="input date")
@click.option("--need_download", type=int, default=1, help="input date")
def main(partition_dt, test_local, need_download):
    if not test_local:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.appName("lyj_test_distribute").enableHiveSupport().getOrCreate()
        spark.conf.set("spark.sql.execution.arrow.enabled",
                       "true")  # accelerate transformation between spark'DataFrame and pandas'DataFrame
        print("job start...")
        print(spark.sparkContext.getConf().getAll())
        unzip_job()  # first unzip JOB.zip for model file
        t1 = datetime.now()
        job_main(spark, partition_dt, test_local, need_download)
        t2 = datetime.now()
        print("job end, cost {} seconds".format((t2 - t1).seconds))
        spark.stop()
    else:
        print("job start...")
        spark = None
        t1 = datetime.now()
        job_main(spark, partition_dt, test_local, need_download)
        t2 = datetime.now()
        print("job end, cost {} seconds".format((t2 - t1).seconds))


if __name__ == "__main__":
    main()

