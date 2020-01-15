# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import click

from model_helper.hyper_parameter_tuning import distributed_param_search

from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
import os
import zipfile


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
    data, targets = make_classification(
        n_samples=10000,
        n_features=300,
        n_informative=12,
        n_redundant=7,
        random_state=134985745,
    )
    clf = DecisionTreeClassifier()
    param_grid = {"max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  "min_samples_leaf": [1, 10, 100, 200, 300],
                  "criterion": ["gini", "entropy"]
                  }
    print("make data size", data.shape)
    data = pd.DataFrame(data, columns=[f"f{i}" for i in range(data.shape[1])])
    #####  distributed param search
    print("test1 test create valid...")
    a = distributed_param_search(spark, data, targets, clf, param_grid, create_valid=True, valid_ratio=0.2,
                                 k_fold=None, random_state=6, broadcast_variable=True, metric_func=roc_auc_score)
    print(a)
    print("test1 done.")

    print("test2 test k_fold...")
    a = distributed_param_search(spark, data, targets, clf, param_grid,
                                 k_fold=3, random_state=6, broadcast_variable=True, metric_func=roc_auc_score)
    print(a)
    print("test2 done.")

    print("test3 test self-defined valid...")
    a = distributed_param_search(spark, data, targets, clf, param_grid, valid_x=data[:1000], valid_y=targets[:1000],
                                 k_fold=None, random_state=6, broadcast_variable=True, metric_func=roc_auc_score)
    print(a)
    print("test3 done.")

    print("test4 test method random...")
    a = distributed_param_search(spark, data, targets, clf, param_grid, method="random", max_iter=10,
                                 k_fold=3, random_state=6, broadcast_variable=True, metric_func=roc_auc_score)
    print(a)
    print("test4 done.")

    print("test5 test valid_set_param...")

    def _update(model, param):
        if param is None:
            return model.get_params()
        else:
            param["n_estimators"] = model.best_iteration
        return param
    valid_set_param = {"model_fit_param": {"eval_metric": "auc", "verbose": False, "early_stopping_rounds": 5},
                       "set_eval_set": True,
                       "update_param_func": _update}
    from xgboost import XGBClassifier
    clf = XGBClassifier()
    a = distributed_param_search(spark, data, targets, clf, param_grid, create_valid=True, valid_ratio=0.2,
                                 k_fold=None, random_state=6, broadcast_variable=True, metric_func=roc_auc_score,
                                 valid_set_param=valid_set_param)
    print(a)
    print("test5 done.")

    print("test6 test cross_val_param...")
    cross_val_param = {"scoring": lambda clf, X, y: roc_auc_score(y_true=y, y_score=clf.predict_proba(X)[:, 1]),
                       "n_jobs": None}
    a = distributed_param_search(spark, data, targets, clf, param_grid,
                                 k_fold=3, random_state=6, broadcast_variable=True,
                                 cross_val_param=cross_val_param)
    print(a)
    print("test6 done.")

    print("test7 test do not broadcast variable with valid set...")
    a = distributed_param_search(spark, data, targets, clf, param_grid, method="grid", create_valid=True, valid_ratio=0.2,
                                 k_fold=None, random_state=6, broadcast_variable=False, save_method="pandas",
                                 metric_func=roc_auc_score)
    print(a)
    print("test7 done.")

    print("test8 test do not broadcast variable with k_fold save_method pandas...")
    a = distributed_param_search(spark, data, targets, clf, param_grid, method="grid",
                                 k_fold=5, random_state=6, broadcast_variable=False, save_method="pandas",
                                 metric_func=roc_auc_score)
    print(a)
    print("test8 done.")

    print("test9 test do not broadcast variable with k_fold save_method sparse...")
    a = distributed_param_search(spark, data, targets, clf, param_grid, method="grid",
                                 k_fold=5, random_state=6, broadcast_variable=False, save_method="sparse",
                                 metric_func=roc_auc_score)
    print(a)
    print("test9 done.")

    print("test10 test do not broadcast variable with k_fold save_method numpy...")
    a = distributed_param_search(spark, data, targets, clf, param_grid, method="grid",
                                 k_fold=5, random_state=6, broadcast_variable=False, save_method="numpy",
                                 metric_func=roc_auc_score)
    print(a)
    print("test10 done.")


def job_main(spark, partition_dt, test_local, need_download):
    """
    任务的流程分为三部分,首先拉取Hive数据至本地,然后处理本地数据,之后将结果数据输出至Hive表
    :param spark: spark session
    :param partition_dt: str,格式%Y%m%d
    :param test_local: 取值0或1(default 0),0表示在DP上执行，1表示本地运行
    :param need_download: 取值0或1(default 1),0表示不需要从hdfs拉取数据，1需要从hdfs拉取数据
    :return: None
    """
    # 1.拉取hive表数据至本地
    # print(f"pull table {INPUT_TABLE} partition_dt={partition_dt} to local...")
    # t1 = datetime.now()
    # 通过表获取数据
    # local_file_path = pull_hive_to_local(spark, hive_table_name=INPUT_TABLE,
    #                                      sql=None, partition_dt=partition_dt,
    #                                      test_local=test_local, need_download=need_download)
    # local_file_path = "empty test"
    # 或者通过SQL获取数据
    # local_file_path = pull_hive_to_local(spark, sql=SQL, partition_dt=partition_dt)
    # t2 = datetime.now()
    # print(f"pull table {INPUT_TABLE} partition_dt={partition_dt} to local succ, cost {(t2-t1).seconds} seconds.")
    # 2.处理获取的数据
    print("process file...")
    t1 = datetime.now()
    df_result = process_file(spark, local_file_path=None)
    t2 = datetime.now()
    print(f"process file succ, cost {(t2-t1).seconds} seconds.")
    # 3.推送数据至hdfs
    # if not test_local:
    #     print(f"push file to table {OUTPUT_TABLE} partition_dt={partition_dt}...")
    #     t1 = datetime.now()
    #     push_file_to_hive_table(spark, df_result, OUTPUT_TABLE, partition_dt)
    #     t2 = datetime.now()
    #     print(f"push file to table {OUTPUT_TABLE} partition_dt={partition_dt} succ, cost {(t2-t1).seconds} seconds.")
    # else:
    #     print("no need to push file to hdfs as param test_local set True")


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
        # partition_dt = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
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
