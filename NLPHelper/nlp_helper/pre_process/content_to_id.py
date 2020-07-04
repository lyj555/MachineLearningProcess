# -*- coding: utf-8 -*-

import numpy as np
from collections import Iterable
from itertools import islice

from ..utils import check_tokenizer
from ..config import PAD, UNK


def _deal_sentence_with_fixed_len(sentence, tokenizer, seq_len, pad_index):
    """
    将一句话进行分割，按照指定长度进行输出，如果不足输出，就补pad_index
    :param sentence: str, 文本内容
    :param tokenizer: callable object,
    :param seq_len: int, 指定的文本长度
    :param pad_index: int, PAD字符的index
    :return: (int, list), 第一个元素表示输入文本的真实长度，如果截断就是截断长度，第二个元素表示划分后文本的内容
    """
    tokens = tokenizer(sentence)
    real_seq_len = len(tokens)
    if real_seq_len < seq_len:
        tokens.extend([pad_index]*(seq_len - real_seq_len))
    else:
        tokens = tokens[: seq_len]
        real_seq_len = seq_len
    return real_seq_len, tokens


def _token_to_number(tokens, real_seq_len, vocab_dic, unk_index):
    """
    将token的列表转化为词汇表的index
    :param tokens: list, 表示一句话的tokens
    :param real_seq_len: int, 这句话的真实长度
    :param vocab_dic: dict, 词汇表字典
    :param unk_index: int, UNK字符的index
    :return: list[int]
    """
    for i in range(real_seq_len):
        tokens[i] = vocab_dic.get(tokens[i], unk_index)
    return tokens


def __parse_content(content, is_file, line_sep, content_index):
    try:
        if isinstance(content, str):
            content = content.strip()
        if not is_file or line_sep is None:  # means only one column
            return content, None, 1
        else:
            parts = content.split(line_sep)
            parts_num = len(parts)
            if parts_num > 2 or parts_num == 1:   # not valid content
                return None, None, parts_num
            else:
                content, label = parts[content_index], parts[1 - content_index]
                return content, label, parts_num
    except:
        return None, None, None


def _check_label(label, label_func=None):
    if label is None: return
    if label_func is None:
        label_func = lambda x: int(x)
    assert callable(label_func), "label_func must be a callable function"
    try:
        label = label_func(label)
    except:
        label = None
    return label


def content_to_id(filepath_or_iter, tokenizer, seq_len, vocab_dic, line_sep="",
                  content_index=0, contain_header=False, with_real_seq_len=False, with_label=True, label_func=None):
    """
    文本内容可以存放于文本中，也可以存在一个迭代器中，如果存放在文件中，
    通过读取文件内容，将其转化为id，要求文件至多包含两列，一列为文本内容，一列为标签，至少包含一列；
    如果存放与迭代器中，要求只能包含文本内容
    :param filepath_or_iter: str or iterable, 文件路径或者是一个迭代的对象
    :param line_sep: line_sep: str or None, 数据分隔符，如果为""
    :param tokenizer: str or callable, 切分器，将一句话切分为多部分，如果是str，有两个选项，word和char
    :param seq_len: int, 指定文本长度，如果文本长度大于该长度，会截断，如果小于该长度，则补PAD
    :param vocab_dic: dict, 词汇表字典，key 是词汇，value是index
    :param content_index: int, 如果输入有两列，指定哪一列是文本列，默认为0，即第一列为文本列
    :param contain_header: bool, 是否包含header，默认为False
    :param with_real_seq_len: bool, 返回结果中是否包含序列的长度，默认为False
    :param with_label: bool, 返回的结果中是否包含真实标签，默认为True
    :param label_func: None or callable, 用于处理label(仅当with_label为True生效)，如果为None，会将label转换为int。
    :return: list[tuple], tuple 根据参数的不同包含不同的返回结果
    """
    if isinstance(filepath_or_iter, str):   # means it's a filepath like
        assert line_sep is None or (isinstance(line_sep, str) and line_sep != ""), \
            f"line separator({line_sep}) must None(means only one column) or valid str, can not be empty."
        assert isinstance(contain_header, bool), f"param contain_header must be bool type"
        assert isinstance(content_index, int) and 0 <= content_index <= 1, \
            f"content_index must be 0 or 1, represents first column or second column is content"

        content_scanner = open(filepath_or_iter, 'r', encoding='UTF-8')
        content_scanner = enumerate(islice(content_scanner, contain_header, None))
        is_file = True
    else:
        assert isinstance(filepath_or_iter, Iterable), \
            f"if filepath_or_iter is not valid path, it must be Iterable object."
        content_scanner = enumerate(filepath_or_iter)
        is_file, with_label = False, False

    tokenizer = check_tokenizer(tokenizer)
    ret_tokens, ret_label, ret_real_seq_len = [], [], []
    bad_line_num, bad_line = 0, []
    for line_number, content in content_scanner:
        content, label, parts_num = __parse_content(content, is_file, line_sep, content_index)
        if with_label:
            label = _check_label(label, label_func)
        if content is None or (is_file and label is None):
            ind = line_number+1+contain_header if is_file else line_number
            if content is None:
                print(f"line num {ind} contain {parts_num} parts,"
                      f"if line_sep is not None, each line must contain two parts, content and label")
            else:
                print(f"line num {ind} contain bad label.")
            bad_line_num += 1
            bad_line.append(ind)
            tokens, real_seq_len = None, None
        else:
            real_seq_len, tokens = _deal_sentence_with_fixed_len(sentence=content, tokenizer=tokenizer,
                                                                 seq_len=seq_len, pad_index=vocab_dic.get(PAD))
            _token_to_number(tokens=tokens, real_seq_len=real_seq_len,
                             vocab_dic=vocab_dic, unk_index=vocab_dic.get(UNK))

        # construct result
        if (content is None or label is None) and with_label and is_file:  # 此时可以同时删除文本和标签，否则保留
            continue
        ret_tokens.append(tokens)
        if with_label:
            ret_label.append(label)
        if with_real_seq_len:
            ret_real_seq_len.append(real_seq_len)

    if bad_line_num != 0:
        print(f"process success, vocab size is {len(vocab_dic)}, "
              f"with bad line number is {bad_line_num}, whose number list is {bad_line}")

    if with_label and with_real_seq_len:
        return np.array(ret_tokens), np.array(ret_label), np.array(ret_real_seq_len)
    elif with_label and not with_real_seq_len:
        return np.array(ret_tokens), np.array(ret_label)
    elif not with_label and with_real_seq_len:
        return np.array(ret_tokens), np.array(ret_real_seq_len)
    else:
        return np.array(ret_tokens)
