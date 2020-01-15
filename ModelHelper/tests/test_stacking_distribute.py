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
from sklearn.linear_model import LogisticRegression, LinearRegression

from model_helper.model_ensemble import StackingClassifier, StackingRegressor

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
    X, y = make_classification(n_samples=5000, n_features=20, n_classes=2, random_state=234)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)

    # test1 test normal process
    print("TEST StackingClassifier...")
    print("test1 test normal process...")
    clf = StackingClassifier(base_learner_list=[RandomForestClassifier(), GradientBoostingClassifier(),
                                                DecisionTreeClassifier()],
                             meta_learner=LogisticRegression(), metric_func=roc_auc_score,
                             distribute=True, spark=spark)
    clf.fit(X=train_x, y=train_y)

    pred = clf.predict_proba(X=test_x)[:, 1]
    auc_val = roc_auc_score(y_true=test_y, y_score=pred)
    print(f"test1 done, auc_val is {auc_val}.")

    # test2 test model selector
    print("test2 test model selector...")

    def selector(model_metrics):
        # print(model_metrics)
        model_avg_metric = np.array(list(map(lambda x: sum(x) / len(x), model_metrics)))
        return model_avg_metric.argsort()[-2:][::-1]  # get top 2 best model

    clf = StackingClassifier(base_learner_list=[RandomForestClassifier(), GradientBoostingClassifier(),
                                                DecisionTreeClassifier()],
                             meta_learner=LogisticRegression(), metric_func=roc_auc_score, select_base_learner=selector,
                             distribute=True, spark=spark)
    clf.fit(X=train_x, y=train_y)

    pred = clf.predict_proba(X=test_x)[:, 1]
    auc_val = roc_auc_score(y_true=test_y, y_score=pred)
    print(f"test2 done, auc_val is {auc_val}.")

    # test3 random select feature in base learner
    print("test3 random select feature in base learner...")

    def selector(model_metrics):
        # print(model_metrics)
        model_avg_metric = np.array(list(map(lambda x: sum(x) / len(x), model_metrics)))
        return model_avg_metric.argsort()[-2:][::-1]  # get top 2 best model

    clf = StackingClassifier(base_learner_list=[RandomForestClassifier(), GradientBoostingClassifier(),
                                                DecisionTreeClassifier()],
                             meta_learner=LogisticRegression(), metric_func=roc_auc_score, select_base_learner=selector,
                             feature_fraction=0.8, distribute=True, spark=spark)
    clf.fit(X=train_x, y=train_y)

    pred = clf.predict_proba(X=test_x)[:, 1]
    auc_val = roc_auc_score(y_true=test_y, y_score=pred)
    print(f"test3 done, auc_val is {auc_val}.")

    # test4 data validation X,y
    print("test4 train&predict...")

    def selector(model_metrics):
        # print(model_metrics)
        model_avg_metric = np.array(list(map(lambda x: sum(x) / len(x), model_metrics)))
        return model_avg_metric.argsort()[-2:][::-1]  # get top 2 best model

    clf = StackingClassifier(base_learner_list=[RandomForestClassifier(), GradientBoostingClassifier(),
                                                DecisionTreeClassifier()],
                             meta_learner=LogisticRegression(), metric_func=roc_auc_score, select_base_learner=selector,
                             feature_fraction=0.8, distribute=True, spark=spark)
    train_x = pd.DataFrame(train_x, columns=[f"f{i}" for i in range(train_x.shape[1])])
    clf.fit(X=train_x, y=train_y)

    print("train by DF, predict numpy array.")
    pred = clf.predict_proba(test_x)[:, 1]  # train by DF, predict array, yes
    auc_val = roc_auc_score(y_true=test_y, y_score=pred)
    print(f"auc val is {auc_val}")

    print("train by DF, predict DF")
    pred = clf.predict_proba(pd.DataFrame(test_x, columns=[f"f{i}" for i in range(test_x.shape[1])]))[:, 1]
    auc_val = roc_auc_score(y_true=test_y, y_score=pred)
    print(f"auc val is {auc_val}")

    clf = StackingClassifier(base_learner_list=[RandomForestClassifier(), GradientBoostingClassifier(),
                                                DecisionTreeClassifier()],
                             meta_learner=LogisticRegression(), metric_func=roc_auc_score, select_base_learner=selector,
                             feature_fraction=0.8, distribute=True, spark=spark)
    # train_x = pd.DataFrame(train_x, columns=[f"f{i}" for i in range(train_x.shape[1])])
    clf.fit(X=np.array(train_x), y=train_y)
    print("train by numpy array, predict numpy array.")
    pred = clf.predict_proba(test_x)[:, 1]  # train by DF, predict array, yes
    auc_val = roc_auc_score(y_true=test_y, y_score=pred)
    print(f"auc val is {auc_val}")

    print("train by numpy array, predict DF")
    pred = clf.predict_proba(pd.DataFrame(test_x, columns=[f"f{i}" for i in range(test_x.shape[1])]))[:, 1]
    auc_val = roc_auc_score(y_true=test_y, y_score=pred)
    print(f"auc val is {auc_val}")
    print("test4 done.")

    # test stratify split data
    print("test5 test stratify split data")
    clf = StackingClassifier(base_learner_list=[RandomForestClassifier(), GradientBoostingClassifier(),
                                                DecisionTreeClassifier()],
                             meta_learner=LogisticRegression(), metric_func=roc_auc_score, select_base_learner=selector,
                             feature_fraction=0.8, distribute=True, spark=spark)
    train_x = pd.DataFrame(train_x, columns=[f"f{i}" for i in range(train_x.shape[1])])
    clf.fit(X=train_x, y=train_y, stratify=True, stratify_col=train_y)

    pred = clf.predict_proba(test_x)[:, 1]  # train by DF, predict array, yes
    auc_val = roc_auc_score(y_true=test_y, y_score=pred)
    print(f"test5 done, auc_val is {auc_val}.")
    print("TEST StackingClassifier SUCC.")

    print("TEST StackingRegressor...")
    X, y = make_regression(n_samples=5000, n_features=20, random_state=224)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)

    # test1 test normal process
    print("test1 test normal process...")
    clf = StackingRegressor(base_learner_list=[RandomForestRegressor(), GradientBoostingRegressor(),
                                               DecisionTreeRegressor()],
                            meta_learner=LinearRegression(), distribute=True, spark=spark)
    clf.fit(X=train_x, y=train_y)

    pred = clf.predict(X=test_x)
    r2_val = r2_score(y_true=test_y, y_pred=pred)
    print(f"test1 done, r2 val is {r2_val}")

    # test2 test model selector
    print("test2 test model selector...")

    def selector(model_metrics):
        # print(model_metrics)
        model_avg_metric = np.array(list(map(lambda x: sum(x) / len(x), model_metrics)))
        return model_avg_metric.argsort()[:2]  # get top 2 best model

    clf = StackingRegressor(base_learner_list=[RandomForestRegressor(), GradientBoostingRegressor(),
                                               DecisionTreeRegressor()],
                            meta_learner=LinearRegression(),
                            metric_func=lambda y_true, y_score: r2_score(y_true=y_true, y_pred=y_score),
                            select_base_learner=selector, distribute=True, spark=spark)
    clf.fit(X=train_x, y=train_y)

    pred = clf.predict(X=test_x)
    r2_val = r2_score(y_true=test_y, y_pred=pred)
    print(f"test2 done, r2 val is {r2_val}")

    # test3 test feature fraction
    print("test3 test feature fraction...")

    def selector(model_metrics):
        # print(model_metrics)
        model_avg_metric = np.array(list(map(lambda x: sum(x) / len(x), model_metrics)))
        return model_avg_metric.argsort()[:2]  # get top 2 best model

    clf = StackingRegressor(base_learner_list=[RandomForestRegressor(), GradientBoostingRegressor(),
                                               DecisionTreeRegressor()],
                            meta_learner=LinearRegression(),
                            metric_func=lambda y_true, y_score: r2_score(y_true=y_true, y_pred=y_score),
                            select_base_learner=selector, feature_fraction=0.8,
                            distribute=True, spark=spark)
    clf.fit(X=train_x, y=train_y)

    pred = clf.predict(X=test_x)
    r2_val = r2_score(y_true=test_y, y_pred=pred)
    print(f"test3 done, r2 val is {r2_val}")
    print("TEST StackingRegressor SUCC.")


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



