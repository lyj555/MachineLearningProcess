# -* coding: utf-8 -*-

import numpy as np
import torch


def seed_everything(seed, use_np=True, use_cpu=True, use_gpu=True):
    assert isinstance(seed, int), "input seed must a integer number"
    if use_np:
        np.random.seed(seed)  # 为numpy设置随机数种子
    if use_cpu:
        torch.manual_seed(seed)  # 为CPU设置种子用于生成固定随机数
    if use_gpu:
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # 为所有的GPU设置种子
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
