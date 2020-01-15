# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import click

from model_helper.feature_selection.wrapper import distributed_feature_select

from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
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
    print("make data size", data.shape)
    data = pd.DataFrame(data, columns=[f"f{i}" for i in range(data.shape[1])])
    ##### test distributed feature select
    print("test1: method: `random`, k_fold: 3...")
    clf = DecisionTreeClassifier()
    subset, effect = distributed_feature_select(spark, data, targets, clf, method="random", k_fold=3,
                                                max_iter=10, random_state=666)
    print(effect, subset)
    print("test1 done.")

    print("test2: method: `random`, k_fold: 3 sample 0.8 input is array...")
    subset, effect = distributed_feature_select(spark, np.array(data), targets, clf, method="random", k_fold=3,
                                                metric_func=roc_auc_score, sample=0.8, max_iter=10, random_state=666)
    print(effect, subset)
    print("test2 done.")

    print("test3 method: `random`, create valid")
    subset, effect = distributed_feature_select(spark, data, targets, clf, method="random",
                                                create_valid=True, valid_ratio=0.2,
                                                metric_func=roc_auc_score,
                                                sample=None, max_iter=10, random_state=777)
    print(effect, subset)
    print("test3 done.")

    print("test4 method: `random`, self-defined valid_set...")
    subset, effect = distributed_feature_select(spark, data, targets, clf, method="random",
                                                valid_x=data[:100], valid_y=targets[:100],
                                                metric_func=roc_auc_score,
                                                sample=None, max_iter=10, random_state=777)
    print(effect, subset)
    print("test4 done.")

    print("test4 method: `random`, self-defined valid_set & min_feature...")
    subset, effect = distributed_feature_select(spark, data, targets, clf, method="random",
                                                valid_x=data[:100], valid_y=targets[:100],
                                                metric_func=roc_auc_score,
                                                sample=None, min_feature=50, max_iter=10, random_state=777)
    print(effect, subset)
    print("test4 done.")

    print("test5: valid_set_param...")

    def _update(model, param):
        if param is None and model.best_iteration_ is not None:
            return {"best_iteration": model.best_iteration_}
        elif param is not None:
            return param
        else:
            return None
    valid_set_param = {"model_fit_param": {"eval_metric": "auc", "verbose": False, "early_stopping_rounds": 5},
                       "set_eval_set": True,
                       "update_param_func": _update}
    clf = LGBMClassifier()
    subset, effect, param = distributed_feature_select(spark, data, targets, clf, method="random",
                                                       create_valid=True, valid_ratio=0.2,
                                                       metric_func=roc_auc_score, valid_set_param=valid_set_param,
                                                       sample=None, max_iter=10, random_state=None)
    print(effect, param, subset)
    print("test5 done.")

    print("test6 cross_val_param...")
    cross_val_param = {"scoring": lambda clf, X, y: roc_auc_score(y_true=y, y_score=clf.predict_proba(X)[:, 1]),
                       "n_jobs": None}
    subset, effect = distributed_feature_select(spark, data, targets, clf, method="random",
                                                create_valid=True, valid_ratio=0.2,
                                                metric_func=roc_auc_score, cross_val_param=cross_val_param,
                                                sample=None, max_iter=10, random_state=None)
    print(effect, subset)
    print("test6 done.")

    print("test7 method do not broadcast valid set method=pandas...")
    clf = LGBMClassifier()
    subset, effect = distributed_feature_select(spark, data, targets, clf, method="random",
                                                create_valid=True, valid_ratio=0.2,
                                                metric_func=roc_auc_score,
                                                max_iter=10, random_state=789, save_method="pandas",
                                                broadcast_variable=False,
                                                verbose=True)
    print(effect, subset)
    print("test7 succ")

    print("test8 method do not broadcast k_fold method=pandas...")
    clf = LGBMClassifier()
    subset, effect = distributed_feature_select(spark, data, targets, clf, method="random",
                                               k_fold=5,
                                               metric_func=roc_auc_score,
                                               max_iter=10, random_state=789, save_method="pandas",
                                               broadcast_variable=False,
                                               verbose=True)
    print(effect, subset)
    print("test8 done.")

    print("test9 method do not broadcast valid set method=sparse...")
    clf = LGBMClassifier()
    subset, effect = distributed_feature_select(spark, data, targets, clf, method="random",
                               create_valid=True, valid_ratio=0.2,
                               metric_func=roc_auc_score,
                               max_iter=10, random_state=789, save_method="sparse",
                               broadcast_variable=False,
                               verbose=True)
    print(effect, subset)
    print("test9 done.")

    print("test10 method do not broadcast valid set method=numpy...")
    clf = LGBMClassifier()
    subset, effect = distributed_feature_select(spark, data, targets, clf, method="random",
                               create_valid=True, valid_ratio=0.2,
                               metric_func=roc_auc_score,
                               max_iter=10, random_state=789, save_method="numpy",
                               broadcast_variable=False,
                               verbose=True)
    print(effect, subset)
    print("test10 done.")

    print("test11 test method `weight`")
    subset, effect = distributed_feature_select(spark, data, targets, clf, method="weight",
                               create_valid=True, valid_ratio=0.2,
                               metric_func=roc_auc_score, valid_set_param=None, cross_val_param=None,
                               sample=None, max_iter=10, random_state=9999)
    print(effect, subset)
    print("test11 done.")

    print("test12 test method `top_feat`")
    subset, effect = distributed_feature_select(spark, data, targets, clf, method="top_feat",
                               create_valid=True, valid_ratio=0.2,
                               metric_func=roc_auc_score, valid_set_param=None, top_ratio_list=[0.9, 0.8, 0.7, 0.6],
                               sample=None, max_iter=10, random_state=9999)
    print(effect, subset)
    print("test12 done.")


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
