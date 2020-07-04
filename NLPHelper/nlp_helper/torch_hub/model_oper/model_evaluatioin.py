# -*- coding: utf-8 -*-

from collections import Iterable

import numpy as np
import torch


def _is_data_iter(data_iter):
    if hasattr(data_iter, "__iter__") and hasattr(data_iter, "__len__"):
        return True
    else:
        return False


def _check_evaluate_data(data, with_label):
    assert isinstance(with_label, bool), "param with_label must be bool object"
    if with_label:
        assert isinstance(data, tuple) or _is_data_iter(data), \
            "if evaluate or with_label=True, param data must be tuple or iterator with x,y"
        if isinstance(data, tuple):
            assert len(data) == 2, "when data is tuple, it must contain two part, x and y"
            assert isinstance(data[0], torch.Tensor) and isinstance(data[1], torch.Tensor), \
                "the element in data must be torch.Tensor object"
        else:
            pass
    else:
        assert isinstance(data, torch.Tensor) or _is_data_iter(data), \
            "if predict or with_label=False, param data must be torch.Tensor or iterator only with x"


def _check_slice_batch(slice_batch):
    assert slice_batch is None or callable(slice_batch), "param slice_batch must be None or callable object."
    if slice_batch is None:
        return lambda x: (x[:-1], x[-1], )
    else:
        return slice_batch


def _model_evaluate(model, data, with_label, feval, y_score_processor, count_every_batch, slice_batch, device):
    _check_evaluate_data(data, with_label)
    model.eval()
    with torch.no_grad():
        if _is_data_iter(data):
            if with_label:  # 带有标签
                if feval is not None:  # 带有评估函数，最终返回评估的结果，一个值
                    if count_every_batch:  # 是否是每个batch计算一次feval值，最终返回均值，比如准确率
                        ret = 0
                        for batch in data:
                            batch_x, batch_y = slice_batch(batch)  # 将batch切分为x和y
                            y_score = model(*[i.to(device) for i in batch_x])
                            y_score = y_score if y_score_processor is None else y_score_processor(y_score)
                            ret += feval(y_score, batch_y.to(device))
                        return ret.item() / len(data)
                    else:  # 每个batch进行预测，存储预测结果和预测值，最后计算feval，比如auc
                        y_score, y_true = [], []
                        for batch in data:
                            batch_x, batch_y = slice_batch(batch)
                            tmp_score = model(*[i.to(device) for i in batch_x])
                            tmp_score = tmp_score if y_score_processor is None else y_score_processor(tmp_score)
                            y_score.append(tmp_score)
                            y_true.append(batch_y.to(device))
                        return feval(torch.cat(y_score, dim=0).to(device), torch.cat(y_true, dim=0).to(device))
                else:  # 未带有评估函数，意味着只预测，返回预测结果和标签，两个值
                    y_score, y_true = [], []
                    for batch in data:
                        batch_x, batch_y = slice_batch(batch)
                        tmp_score = model(*[i.to(device) for i in batch_x])
                        tmp_score = tmp_score if y_score_processor is None else y_score_processor(tmp_score)

                        # y_score = np.append(y_score, tmp_score.cpu().numpy())
                        # y_true = np.append(y_true, batch_y.cpu().numpy())
                        y_score.extend(tmp_score.cpu().numpy().tolist())
                        y_true.extend(batch_y.cpu().numpy().tolist())
                    return y_score, y_true
            else:
                # 忽略feval的取值，仅仅预测，返回预测结果，一个值
                y_score = np.array([])
                for batch in data:
                    batch_x, _ = slice_batch(batch)
                    tmp_score = model(*[i.to(device) for i in batch_x])
                    tmp_score = tmp_score if y_score_processor is None else y_score_processor(tmp_score)
                    y_score = np.append(y_score, tmp_score.cpu().numpy())
                return y_score
        else:  # 不是iterator，是tuple or Tensor
            if with_label:  # 带有标签，此时数据必须是tuple，且包含两部分x和y
                x, y = slice_batch(data)
                if feval is not None:  # 带有评估函数，最终返回评估的结果，一个值
                    y_score = model(*[i.to(device) for i in x])
                    y_score = y_score if y_score_processor is None else y_score_processor(y_score)
                    return feval(y_score, y)
                else:  # 未带有评估函数，意味着只预测，返回预测结果和标签，两个值（这个分支的作用不大）
                    y_score = model(*[i.to(device) for i in x])
                    return y_score.cpu().numpy(), y.cpu.numpy()
            else:
                # 忽略feval的取值，未带标签，仅仅返回预测结果，一个变量
                data = slice_batch(data)
                y_score = model(*[i.to(device) for i in data])
                y_score = y_score if y_score_processor is None else y_score_processor(y_score)
                return y_score


def evaluate(model, data, feval=None, count_every_batch=False, y_score_processor=None, slice_batch=None, device=None):
    """
    模型效果评估，输入的数据部分需要包含数据和标签两部分
    :param model: torch.nn.Module, torch model
    :param data: iterator or tuple(x, y), 输入数据
    :param feval: None or callable, 如果是None，不进行评估，返回预测值和标签，否则进行评估，返回一个评估值.
    feval的输入为两部分，一部分是y_score, 第二部分是y_true, 均为Tensor类型，返回值需要为一个值，可以是tensor或者
    :param count_every_batch: bool, 只有feval不为None时生效，如果为True，每次batch迭代会计算一次feval值，需要为Tensor
    最后取所有feval值的均值，如果为False，记录每次的预测值，最终计算feval
    :param y_score_processor: None or callable, 对model(x)的预测结果进行加工，如果为None不加工，否则进行处理，
    其输入为一部分。比如预测为n行多列，想将其压缩为n行1列，则需要通过此进行处理
    :param slice_batch: None or callable, 对输入batch进行训练数据和标签数据的切分，必须返回两部分，
        默认为None, 表示lambda x: (x[:-1], x[-1], )
    :param device: None or callable, 转换数据类型
    :return:
    """
    assert feval is None or callable(feval), "feval must be None or callable object!"
    assert y_score_processor is None or callable(y_score_processor), "y_score_processor must be None or callable object"
    assert device is None or isinstance(device, torch.device), "device must be None or torch.device"
    slice_batch = _check_slice_batch(slice_batch)
    evaluate_ret = _model_evaluate(model, data, feval=feval, count_every_batch=count_every_batch,
                                   y_score_processor=y_score_processor, device=device, with_label=True,
                                   slice_batch=slice_batch)
    return evaluate_ret


def predict(model, data, with_label=False, y_score_processor=None, slice_batch=None, device=None):
    """
    模型预测
    :param model: torch.nn.Module, torch model
    :param data: iterator or x, 输入数据
    :param with_label: 是否带有标签
    :param y_score_processor: None or callable, 对model(x)的预测结果进行加工，如果为None不加工，否则进行处理，
    其输入为一部分。比如预测为n行多列，想将其压缩为n行1列，则需要通过此进行处理
    :param slice_batch: None or callable, 对输入batch进行训练数据和标签数据的切分，必须返回两部分，
        默认为None, 表示lambda x: (x[:-1], x[-1], )
    :param device: None or callable, 转换数据类型
    :return:
    """
    assert isinstance(model, torch.nn.Module), "model must torch.nn.Module object"
    assert y_score_processor is None or callable(y_score_processor), "y_score_processor must be None or callable object"
    assert device is None or isinstance(device, torch.device), "device must be None or torch.device"
    slice_batch = _check_slice_batch(slice_batch)
    predict_ret = _model_evaluate(model, data, device=device, y_score_processor=y_score_processor,
                                  slice_batch=slice_batch, with_label=with_label,
                                  feval=None, count_every_batch=False)
    return predict_ret
