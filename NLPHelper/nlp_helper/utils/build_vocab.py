# -*- coding: utf-8 -*-

from itertools import islice
from collections import Iterable
import pickle

from .check_tokenizer import check_tokenizer
from ..config import PAD, UNK


def _file_to_word_count_dic(file_path, line_sep, tokenizer, content_index=0, contain_header=False):
    """
    对于指定的文件，将其变为word count，形如{"word1": 10, "word2": 20, ...}
    :param file_path: str, 文件路径
    :param line_sep: str or None, 数据分隔符，如果为None，表示只有内容这一列
    :param content_index: int, 如果数据存在分隔符，那么内容的index是哪一部分，从零开始计数
    :param contain_header: bool, 是否包含header
    :param tokenizer: callable object, 将一句话分割为多个部分，返回一个tuple
    :return: dict, key is word, value is the number of the word
    """
    assert line_sep is None or isinstance(line_sep, str), f"line separator {line_sep} must None or str"
    if line_sep is not None:
        assert isinstance(content_index, int), f"if line sep is not None, you must specify the content column index"
    assert isinstance(contain_header, bool), f"param contain_header must be bool type"

    if isinstance(file_path, str):
        vocab_count_dic = {}
        bad_line_num, bad_line = 0, []
        with open(file_path, 'r', encoding='UTF-8') as f:
            for line_number, line in enumerate(islice(f, contain_header, None)):
                try:
                    lin = line.strip()
                    if line_sep is None:  # means only one column
                        content = lin
                    else:
                        content = lin.split(line_sep)[content_index]

                    for word in tokenizer(content):
                        vocab_count_dic[word] = vocab_count_dic.get(word, 0) + 1
                except:
                    bad_line_num += 1
                    bad_line.append(line_number+1+contain_header)
        if bad_line_num > 0:
            print(f"with bad line number is {bad_line_num}, whose number list is {bad_line}")
    elif isinstance(file_path, Iterable):
        vocab_count_dic = {}
        bad_line_num, bad_line = 0, []
        for line_number, line in enumerate(file_path):
            try:
                for word in tokenizer(line):
                    vocab_count_dic[word] = vocab_count_dic.get(word, 0) + 1
            except:
                bad_line_num += 1
                bad_line.append(line_number + 1 + contain_header)
        if bad_line_num > 0:
            print(f"with bad line number is {bad_line_num}, whose number list is {bad_line}")
    else:
        raise ValueError("param file_path is not valid")

    return vocab_count_dic


def _filter_vocab(vocab_dic, max_vocab_size=None, min_word_freq=1):
    """
    filter vocabulary by specified maximum vocab_size and minimum word frequency
    :param vocab_dic: dict, key is word, value is count number
    :param max_vocab_size: int or None, if None, all word will be treat as vocab word, default is None
    :param min_word_freq: int, minimum word frequency , default is 1
    :return:
    """
    assert min_word_freq >= 1, f"param min_word_freq {min_word_freq} must greater than or equal to 1"
    assert max_vocab_size is None or max_vocab_size >= 1, f"param max_vocab_size {max_vocab_size} can be None or >= 1"

    if min_word_freq == 1 and max_vocab_size is None:  # no need filter
        return list(vocab_dic.keys())

    if max_vocab_size is not None:
        if min_word_freq == 1:
            filtered_word = vocab_dic.items()
        else:
            filtered_word = []
            for word, word_count in vocab_dic.items():
                if word_count >= min_word_freq:
                    filtered_word.append((word, word_count))
        filtered_word = sorted(filtered_word, key=lambda x: x[1], reverse=True)[:max_vocab_size]
        return [i[0] for i in filtered_word]
    else:
        filtered_word = []
        for word, word_count in vocab_dic.items():
            if word_count >= min_word_freq:
                filtered_word.append(word)
        return filtered_word


def _generate_word_dic(word_list, add_pad_unk=True, vocab_dic=None):
    """
    generate word dictionary by word_list, index 0 is PAD, index len(word_list)+1 is UNK
    :param word_list: list or str, word list, means it is file-like path, each line is a word
    :param add_pad_unk: bool, if or not add PAD and UNK into dict, default 0 is is PAD, len(word_dic)+1 is UNK
    :return: dict, key is word, value is
    """
    assert isinstance(word_list, list) or isinstance(word_list, str), \
        f"unknown type({type(word_list)}) for word list, expect list or str"
    assert vocab_dic is None or isinstance(vocab_dic, dict), "input vocab_dic must be None or dict object"

    if vocab_dic is None:
        vocab_dic = {}
    else:
        vocab_dic = vocab_dic.copy()

    start_ind = len(vocab_dic) if PAD in vocab_dic and vocab_dic[PAD] == 0 else len(vocab_dic)+1
    if isinstance(word_list, list):
        for word in word_list:
            if word not in vocab_dic:
                vocab_dic[word] = start_ind
                start_ind += 1
    else:
        with open(word_list, "r") as f:
            for word in f:
                word = word.strip()
                if word not in vocab_dic:
                    vocab_dic[word] = start_ind
                    start_ind += 1

    if add_pad_unk and PAD not in vocab_dic and UNK not in vocab_dic:
        vocab_dic[PAD] = 0
        vocab_dic[UNK] = len(vocab_dic)

    return vocab_dic


def build_vocab_by_raw_file(file_path, line_sep, tokenizer, content_index=0, contain_header=False, vocab_dic=None,
                            max_vocab_size=None, min_word_freq=1, add_pad_unk=True, word_dic_save_path=None):
    """
    通过原始文本构建词汇表的字典，key值是word，value是index，其中0代表<PAD>，所有词汇数+1表示<UNK>
    :param file_path: str, 原始文本的文件路径
    :param line_sep: str or None, 数据分隔符，如果为None，表示输入的文件中只有内容这一列
    :param tokenizer: callable object, 一个函数，将一句话分割为多个部分，返回一个tuple
    :param content_index: int, 如果数据存在分隔符，指定文本数据是第几列，从零开始计数，默认为0
    :param contain_header: contain_header: bool, 输入的文本文件中是否包含header， 默认为False
    :param vocab_dic: None or dict, 是否指定原始字典词汇表，如果指定，会基于此进行更新
    :param max_vocab_size: max_vocab_size: int or None, 最大允许的此表数量，默认为None，不做限制，
    如果为int，会按照词汇数据量排序去top n
    :param min_word_freq: min_word_freq: int, 词汇最低出现的次数，默认为1
    :param add_pad_unk: bool, 是否添加PAD和UNK字符到词汇字典中，默认为True
    :param word_dic_save_path: None or str, 构建词汇表的保存路径，如果为None，那么不保存
    :return: dict, key is word, value is index
    """
    assert word_dic_save_path is None or isinstance(word_dic_save_path, str), \
        f"param word_dic_save_path must be None file-like path"
    tokenizer = check_tokenizer(tokenizer)
    word_count_dic = _file_to_word_count_dic(file_path=file_path, line_sep=line_sep, tokenizer=tokenizer,
                                             content_index=content_index, contain_header=contain_header)
    word_list = _filter_vocab(vocab_dic=word_count_dic, max_vocab_size=max_vocab_size, min_word_freq=min_word_freq)
    word_index_dic = _generate_word_dic(word_list=word_list, add_pad_unk=add_pad_unk, vocab_dic=vocab_dic)
    print(f"vocabulary dict built success, size is {len(word_index_dic)}")
    if word_dic_save_path is not None:
        pickle.dump(word_index_dic, open(word_dic_save_path, 'wb'))
        print(f"save word dic success, path is {word_dic_save_path}.")
    return word_index_dic


def build_vocab_by_word_file(file_path, add_pad_unk=True, word_dic_save_path=None, vocab_dic=None):
    """
    根据词汇表的文件（每行是一个词汇）构建词汇表字典，key值是word，value是index，其中0代表<PAD>，所有词汇数+1表示<UNK>
    :param file_path: str, 词汇表文件的路径
    :param word_dic_save_path: None or str, 构建词汇表的保存路径，如果为None，那么不保存
    :param add_pad_unk: bool, 是否添加PAD和UNK字符到词汇字典中，默认为True
    :param vocab_dic: None or dict, 是否指定原始字典词汇表，如果指定，会基于此进行更新
    :return: dict, key is word, value is index
    """
    word_index_dic = _generate_word_dic(file_path, add_pad_unk=add_pad_unk, vocab_dic=vocab_dic)
    print(f"vocabulary dict built success, size is {len(word_index_dic)}")
    if word_dic_save_path is not None:
        pickle.dump(word_index_dic, open(word_dic_save_path, 'wb'))
        print(f"save word dic success, path is {word_dic_save_path}.")
    return word_index_dic
