# -*- coding: utf-8 -*-

import tqdm
import pickle

UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def _build_vocab(file_path, tokenizer, max_vocab_size, min_word_freq, line_sep):
    vocab_dic = {}
    bad_line = 0
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                bad_line += 1
                continue

            seps = lin.split(line_sep)
            if 1 <= len(seps) <= 2:
                content = seps[0]  # default first part separated is content, second is label
            else:
                bad_line += 1
                continue

            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_word_freq],
                            key=lambda x: x[1], reverse=True)[:max_vocab_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    print(f"bad line number is {bad_line}")
    return vocab_dic


def _line_to_number(content, tokenizer, vocab_dic, pad_size):
    token = tokenizer(content)
    seq_len = len(token)
    if seq_len < pad_size:
        token.extend([vocab_dic.get(PAD)]*(pad_size - seq_len))
    else:
        token = token[: pad_size]
        seq_len = pad_size

    words_line = []
    for word in token:
        words_line.append(vocab_dic.get(word, vocab_dic.get(UNK)))
    return words_line, seq_len


def format_content(file_path, if_with_label, train_granularity, build_vocab, vocab_save_path,
                   max_vocab_size, min_word_freq, pad_size, line_sep, word_sep=" "):
    if train_granularity == "word":
        tokenizer = lambda x: x.split(word_sep)
    elif train_granularity == "char":
        tokenizer = lambda x: [y for y in x]
    else:
        raise ValueError("param train_granularity must be in ('word', 'char')!")

    if build_vocab:
        vocab_dic = _build_vocab(file_path, tokenizer, max_vocab_size, min_word_freq, line_sep)
        pickle.dump(vocab_dic, open(vocab_save_path, 'wb'))
        print(f"create vocabulary from {file_path}, vocabulary size {len(vocab_dic)}")
    else:
        vocab_dic = pickle.load(open(vocab_save_path, 'rb'))
        print(f"load vocabulary from {vocab_save_path}, vocabulary size {len(vocab_dic)}")

    contents = []
    bad_line = 0
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            seps = lin.split(line_sep)
            if if_with_label:
                if len(seps) == 2:
                    content, label = seps
                    words_line, seq_len = _line_to_number(content, tokenizer, vocab_dic, pad_size)
                    contents.append((words_line, seq_len, int(label)))
                else:
                    bad_line += 1
                    continue
            else:
                if len(seps) == 1:
                    content = seps
                    words_line, seq_len = _line_to_number(content, tokenizer, vocab_dic, pad_size)
                    contents.append((words_line, seq_len, None))
                else:
                    bad_line += 1
                    continue
    print(f"bad line number is {bad_line}")
    return contents

