# -*- coding: utf-8 -*-

from hdfs.client import InsecureClient

from .save_load_train_valid_from_hdfs import save_train_valid_to_hdfs, load_train_valid_to_local
from ..config import WEB_USER, WEB_HDFS_URL, TEMP_HDFS_PATH


def dynamic_confirm_partition_num(sc):
    driver_memory = sc.getConf().get('spark.driver.memory')
    driver_cores = sc.getConf().get('spark.driver.cores')

    if_dynamic_allocation = sc.getConf().get('spark.dynamicAllocation.enabled')
    min_executors = sc.getConf().get('spark.dynamicAllocation.minExecutors')
    max_executors = sc.getConf().get('spark.dynamicAllocation.maxExecutors')
    num_executors = sc.getConf().get('spark.executor.instances')
    executor_memory = sc.getConf().get('spark.executor.memory')
    executor_cores = sc.getConf().get('spark.executor.cores')
    print(f"driver info: driver memory is {driver_memory}, "
          f"driver core num is {driver_cores}")
    print(f"executor info: dynamic allocation set {if_dynamic_allocation},"
          f"min executors {min_executors}, max executors {max_executors},"
          f"num executors {num_executors}, executor memory {executor_memory}, executor cores {executor_cores}")
    if if_dynamic_allocation == "true":
        num_partitions = (int(max_executors)-1)*int(executor_cores)
    elif if_dynamic_allocation == "false":
        if num_executors is not None:
            num_partitions = int(num_executors)*int(executor_cores)
        else:
            num_partitions = 3  # default value
    else:
        num_partitions = 3  # default value
    return num_partitions


def uniform_partition(spark, content_list, num_partition):
    partitioner = [(i+1, content_list[i]) for i in range(len(content_list))]
    # para search
    s = spark.sparkContext.parallelize(partitioner, numSlices=1)
    s = s.partitionBy(num_partition, partitionFunc=lambda x: divmod(x, num_partition)[1])  # x is data set key,[1] is yu
    # check 是否均匀分区
    _ret = [len(i) for i in s.glom().collect()]
    # print(f"max partition num {max(_ret)}, min partition_num {min(_ret)}")
    return s


def save_driver_data(spark, broadcast_variable, train_x, train_y, valid_x, valid_y, save_method):
    if broadcast_variable:
        train_x = spark.sparkContext.broadcast(train_x) if train_x is not None else train_x
        train_y = spark.sparkContext.broadcast(train_y) if train_y is not None else train_y
        valid_x = spark.sparkContext.broadcast(valid_x) if valid_x is not None else valid_x
        valid_y = spark.sparkContext.broadcast(valid_y) if valid_y is not None else valid_y
        hdfs_path_dic = None
    else:
        hdfs = InsecureClient(WEB_HDFS_URL, user=WEB_USER)
        hdfs_path_dic = save_train_valid_to_hdfs(hdfs=hdfs, method=save_method, tmp_hdfs_path=TEMP_HDFS_PATH,
                                                 train_x=train_x, train_y=train_y,
                                                 valid_x=valid_x, valid_y=valid_y)
        train_x, train_y, valid_x, valid_y = (None,) * 4
    return train_x, train_y, valid_x, valid_y, hdfs_path_dic


def load_driver_data(hdfs_path, train_x, train_y, valid_x, valid_y, save_method):
    if hdfs_path is None:
        from pyspark.broadcast import Broadcast
        train_x = train_x.value if isinstance(train_x, Broadcast) else train_x
        train_y = train_y.value if isinstance(train_y, Broadcast) else train_y
        valid_x = valid_x.value if isinstance(valid_x, Broadcast) else valid_x
        valid_y = valid_y.value if isinstance(valid_y, Broadcast) else valid_y
    else:
        hdfs = InsecureClient(WEB_HDFS_URL, user=WEB_USER)
        data_dic = load_train_valid_to_local(hdfs=hdfs, method=save_method, ret_dic=hdfs_path)
        train_x, train_y, valid_x, valid_y = data_dic["train_x"], data_dic["train_y"], \
                                             data_dic["valid_x"], data_dic["valid_y"]
        del data_dic
    return train_x, train_y, valid_x, valid_y

