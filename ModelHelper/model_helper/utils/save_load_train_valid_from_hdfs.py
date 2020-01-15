# -*- coding: utf-8 -*-

import os
from datetime import datetime
import pandas as pd
import numpy as np
import scipy


def save_train_valid_to_hdfs(hdfs, method, tmp_hdfs_path, **kwargs):
    assert method in ("pandas", "sparse", "numpy"), "param method must be in ('pandas', 'sparse', 'numpy')!"
    suffix = f"{datetime.now().strftime('%Y%m%d%H%M%S')}"
    ret = {}
    for i in kwargs:
        if kwargs[i] is not None:
            if method == "pandas":
                temp_local_path = i + "_" + suffix + ".csv"
                pd.DataFrame(kwargs[i]).to_csv(temp_local_path, index=None, header=False)
            elif method == "numpy":
                temp_local_path = i + "_" + suffix + ".npy"
                np.save(temp_local_path, kwargs[i])
            else:
                temp_local_path = i + "_" + suffix + ".npz"
                sp = scipy.sparse.csr_matrix(kwargs[i])
                scipy.sparse.save_npz(temp_local_path, sp)
            temp_hdfs_path = os.path.join(tmp_hdfs_path, temp_local_path)
            hdfs.upload(local_path=temp_local_path, hdfs_path=temp_hdfs_path)
            ret[i] = temp_hdfs_path
        else:
            ret[i] = None
    return ret


def load_train_valid_to_local(hdfs, ret_dic, method):
    assert method in ("pandas", "sparse", "numpy"), "param method must be in ('pandas', 'sparse', 'numpy')!"
    ret = {}
    for i in ret_dic:
        temp_hdfs_path = ret_dic[i]
        if temp_hdfs_path is not None:
            temp_local_path = os.path.basename(temp_hdfs_path)
            if os.path.exists(temp_local_path):
                print("exists...")
            else:
                hdfs.download(hdfs_path=temp_hdfs_path, local_path=temp_local_path)

            if method == "pandas":
                ret[i] = pd.read_csv(temp_local_path, header=None).values
            elif method == "numpy":
                ret[i] = np.load(temp_local_path)
            else:
                ret[i] = scipy.sparse.load_npz(temp_local_path).toarray()
                ret[i] = ret[i][0] if "y" in i else ret[i]
        else:
            ret[i] = None
    return ret
