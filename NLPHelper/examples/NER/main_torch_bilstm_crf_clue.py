# -*- coding: utf-8 -*-

import os
import json
import pickle
from itertools import zip_longest
from datetime import datetime
from functools import reduce
import torch
import numpy as np

from nlp_helper.torch_hub.models import BiLSTM, CRF
from nlp_helper.torch_hub.data_loader import torch_iterator
from nlp_helper.utils import build_vocab_by_raw_file
from nlp_helper.pre_process import content_to_id
from nlp_helper.torch_hub.model_oper import init_network, SelfModel, seed_everything, evaluate

START_TAG = "[START]"
END_TAG = "[END]"


def _add_line_pos(line_pos, pos_tag, positions):
    for position in positions:
        for start_idx, end_idx, in position:
            assert end_idx >= start_idx, f"end_idx {end_idx} must greater than start_idx {start_idx}"
            assert end_idx < len(line_pos), "end_idx must less than length"
            for i in range(start_idx, end_idx+1):
                if line_pos[i] != "O":
                    return None
                if i == start_idx:
                    line_pos[i] = f"B-{pos_tag}"
                # elif i == end_idx:
                #     line_pos[i] = f"E-{pos_tag}"
                else:
                    line_pos[i] = f"I-{pos_tag}"
    return 1


def _parse_line(line_dic):
    assert "text" in line_dic and "label" in line_dic, "key `text` and `label` must be in line_dic!"
    line_len = len(line_dic["text"])
    line_pos = ["O"]*line_len
    for pos_tag in line_dic["label"]:
        status = _add_line_pos(line_pos, pos_tag, positions=line_dic["label"][pos_tag].values())
        if status is None:
            print(line_dic)
            return None
    return list(line_dic["text"]), line_pos


def construct_data(file_path):
    """
    将输入的数据组装为 sentence tag_list的形式
    :param file_path:
    :return: sample, tag
    """
    sample_list = []
    tag_list = []
    with open(file_path, "r", encoding='utf-8') as f:
        bad_num = 0
        for line in f:
            line = line.strip("\n")
            ret = _parse_line(json.loads(line))
            if ret is not None:
                sample_list.append(ret[0])
                tag_list.append(ret[1])
            else:
                bad_num += 1
        print(bad_num)
    return sample_list, tag_list


def sort_sequence(samples, tags):
    indices = sorted(range(len(samples)), key=lambda x: len(samples[x]), reverse=True)
    return [samples[i] for i in indices], [tags[i] for i in indices]


def raw_data_to_model(file_path, tokenizer, word2id, tag2id, batch_size, contain_y=True):
    sample_list_, tag_list_ = construct_data(file_path)
    sample_list_, tag_list_ = sort_sequence(sample_list_, tag_list_)
    x, y, lengths = [], [], []
    for i in range(0, len(sample_list_), batch_size):  # 每个batch按照最大len进行to id操作
        # seq_len_ = max(map(lambda xx: len(xx), sample_list_[i:i+batch_size]))
        seq_len_ = len(sample_list_[i])
        x_, lengths_ = content_to_id(sample_list_[i:i+batch_size], line_sep=None, tokenizer=tokenizer, seq_len=seq_len_,
                                     vocab_dic=word2id, with_real_seq_len=True)
        if contain_y:
            y_ = content_to_id(tag_list_[i:i+batch_size], line_sep=None, tokenizer=tokenizer, seq_len=seq_len_,
                               vocab_dic=tag2id)
            y.extend(y_.tolist())
        x.extend(x_.tolist())
        lengths.extend(lengths_.tolist())

    if contain_y:
        return np.array(x), np.array(y), np.array(lengths)
    else:
        return np.array(x), np.array(lengths)


class BiLSTM_CRF(torch.nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_tags, start_idx, stop_idx):
        super(BiLSTM_CRF, self).__init__()
        self.bi_lstm = BiLSTM(vocab_size, emb_size, hidden_size)
        self.crf = CRF(2*hidden_size, num_tags, start_idx, stop_idx)

    def forward(self, sents_tensor, lengths):
        features = self.bi_lstm(sents_tensor, lengths)
        crf_scores = self.crf(features)
        return crf_scores


def __add_entity(entity, ret):
    if entity[0] in ret:
        ret[entity[0]].append((entity[1], entity[2]))
    else:
        ret[entity[0]] = [(entity[1], entity[2])]


def id2tag(ids, id2tag_dic):
    return [id2tag_dic[id_] for id_ in ids]


def parse_entity_from_sequence(seq):
    """
    ['B-PER', 'I-PER', 'O', 'B-LOC']  返回 {'PER': [(0, 1)], 'LOC': [(3, 3)]}
    ['B-PER', 'I-PER', 'O', 'B-LOC', 'B-NAME', "I-NAME", "I-Q"]  返回 {'PER': [(0, 1)], 'LOC': [(3, 3)], 'NAME': [(4, 5)]}
    ['B-PER', 'I-PER', 'O', 'B-LOC', 'B-PER', "I-PER", "I-PER", "I-Q"] 返回 {'PER': [(0, 1), (4, 6)], 'LOC': [(3, 3)]}
    :param seq:
    :return:
    """
    ret = {}
    cur = []
    for i in range(len(seq)):
        if seq[i] == "O" or "-" not in seq[i]:
            if len(cur) != 0:
                __add_entity(cur, ret)
                cur = []
        else:
            start, tag = seq[i].split("-")  # 划分为B/I和tag
            if start == "B":
                if len(cur) != 0:  # 意味着之前的需要存储
                    __add_entity(cur, ret)
                    cur = [tag, i, i]  # 开始存储tag, tag起始的index, tag中止的index
                else:
                    cur = [tag, i, i]  # 开始存储tag, tag起始的index, tag中止的index
            else:
                if len(cur) != 0:
                    if tag == cur[0]:
                        cur[2] = i  # 修改tag中止的index
                    else:
                        __add_entity(cur, ret)  # 意味着新的tag出现且I开头，不合法
                        cur = []
                else:  # 意味着新的tag出现且I开头，不合法
                    pass
    if len(cur) != 0:  # 最后的实体添加
        __add_entity(cur, ret)
    return ret


def _evaluate_one_sentence(y_true_entity, y_score_entity):
    """
    y_true_entity: {'PER': [(0, 1), (4, 5)], 'LOC': [(3, 3)]}
    y_score_entity: {'PER': [(0, 1), (5, 6)], 'LOC': [(3, 3), (8, 10)]}
    :param y_true_entity:
    :param y_score_entity:
    :return: (2, 3, 4) 预测正确的实体数量，真实的全部实体数量，预测的全部实体数量
    """
    y_true_entity_num = 0
    for key in y_true_entity:
        y_true_entity_num += len(y_true_entity[key])

    y_score_entity_num = 0
    for key in y_score_entity:
        y_score_entity_num += len(y_score_entity[key])

    correct_num = 0
    for tag, indexes in y_score_entity.items():
        if tag in y_true_entity:
            for index in indexes:
                if index in y_true_entity[tag]:
                    correct_num += 1
        else:
            pass
    return correct_num, y_true_entity_num, y_score_entity_num


def evaluate_all_sentence(y_true, y_score):
    y_true_entity_num, y_score_entity_num, correct_num = 0, 0, 0
    for y_true_, y_score_ in zip(y_true, y_score):
        y_true_ = parse_entity_from_sequence(y_true_)
        y_score_ = parse_entity_from_sequence(y_score_)
        correct_num_, y_true_entity_num_, y_score_entity_num_ = _evaluate_one_sentence(y_true_, y_score_)

        correct_num += correct_num_
        y_true_entity_num += y_true_entity_num_
        y_score_entity_num += y_score_entity_num_
    print(y_true_entity_num, y_score_entity_num, correct_num)
    recall = correct_num/y_true_entity_num
    precision = correct_num/y_score_entity_num
    f1 = 2*recall*precision/(precision+recall)
    return recall, precision, f1


def main_entry():
    save_dir = "./data/ner"
    vocab_file_path = "./data/ner/cluener_public/train.json"
    tokenizer = lambda x: x  # 输入是list, 相当于已经tokenize

    # 1. 构建词典
    sample_list, tag_list = construct_data(vocab_file_path)   # 1 bad line, 实体嵌套实体
    ## 1.1 构建word2id词典
    # word_save_path = os.path.join(save_dir, "train_word_vocab.pkl")
    word2id = build_vocab_by_raw_file(sample_list, line_sep=None, tokenizer=tokenizer)

    ## 1.2 构建tag2id词典
    # tag_save_path = os.path.join(save_dir, "train_tag_crf_vocab.pkl")
    tag2id = build_vocab_by_raw_file(tag_list, line_sep=None, tokenizer=tokenizer)
    tag2id[START_TAG] = len(tag2id)
    tag2id[END_TAG] = len(tag2id)

    # 2. 构造训练、验证和测试数据
    #    构造三部分数据并将其转换为ID
    train_path = "./data/ner/cluener_public/train.json"
    valid_path = "./data/ner/cluener_public/dev.json"
    test_path = "./data/ner/cluener_public/test.json"
    batch_size = 128

    train_x, train_y, train_lengths = raw_data_to_model(train_path, tokenizer, word2id, tag2id, batch_size)
    print(f"train_x sample number is {len(train_x)}, label sample number is {len(train_y)}")

    valid_x, valid_y, valid_lengths = raw_data_to_model(valid_path, tokenizer, word2id, tag2id, batch_size)
    print(f"valid_x sample number is {len(valid_x)}, label sample number is {len(valid_y)}")

    # test_x, test_y, test_lengths = raw_data_to_model(test_path, tokenizer, word2id, tag2id, batch_size)
    # print(f"test_x sample number is {len(test_x)}, label sample number is {len(test_y)}")

    # 3. 转换数据为迭代器
    # batch_size = 128
    # small_sample_test = False
    # small_sample_num = 10000
    # if small_sample_test:
    #     train_x, train_lengths, train_y = train_x[:small_sample_num], train_lengths[:small_sample_num], train_y[:small_sample_num]

    train_iter = torch_iterator(batch_data=(train_x, train_lengths, train_y,), batch_size=batch_size)
    valid_iter = torch_iterator(batch_data=(valid_x, valid_lengths, valid_y,), batch_size=batch_size)
    # test_iter = torch_iterator(batch_data=(test_x, test_lengths, test_y), batch_size=batch_size)

    # 4. 初始化模型
    seed_everything(1024, use_np=True, use_cpu=True, use_gpu=True)

    model = BiLSTM_CRF(vocab_size=len(word2id), emb_size=50, hidden_size=32, num_tags=len(tag2id),
                       start_idx=tag2id[START_TAG], stop_idx=tag2id[END_TAG])
    init_network(model)
    print(model)

    # 4. 模型训练
    num_epochs = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 1e-3
    model_save_path = os.path.join(save_dir, "bilstm_model.pt")
    print("now the device is ", device)

    loss = model.crf.loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    clf = SelfModel(model=model)
    t1 = datetime.now()
    clf.train(train_iter, num_epochs, loss=loss, optimizer=optimizer, valid_iter=valid_iter,
              early_stopping_batch=30, batch_check_frequency=2,
              print_every_batch=3, model_save_path=model_save_path, device=device)
    t2 = datetime.now()
    print(f"train cost {(t2-t1).seconds} seconds")

    # 5. 模型评估
    ## 5.1 解码  涉及到crf的解码，没有使用内置的evaluate，自己调用模型进行预测然后进行解码
    decode = model.crf.viterbi_decode
    id2tag_dic = {id_: tag for tag, id_ in tag2id.items()}
    #  y_score, y_true = evaluate(clf.model, train_iter, y_score_processor=get_max_prob_index)
    y_score, y_true = [], []
    for sent, leng, y_true_ in valid_iter:
        y_true_ = y_true_.cpu()
        crf_score = clf.model(sent.to(device), leng.to(device))
        y_score_tag = decode(crf_score.cpu(), sent.gt(0).cpu())[1]

        lengs = leng.cpu().numpy()
        for i in range(len(lengs)):  # 遍历样本
            y_score.append(id2tag(y_score_tag[i][:lengs[i]], id2tag_dic))
            y_true.append(id2tag(y_true_[i][:lengs[i]].numpy(), id2tag_dic))

    ## 5.2 评估指标
    metrices = evaluate_all_sentence(y_true, y_score)
    print(metrices)
    # 3072 2909 1944
    # (0.6328125, 0.6682708834651083, 0.6500585186423675)

    # 6. 预测
    # 对测试集进行预测，然后将格式整理为cluemark的格式，提交到线上查看效果
    with open(test_path, "r") as f:
        y_score = []
        for line in f:
            line = line.strip("\n")
            line_text = json.loads(line)["text"]
            sent, leng = content_to_id([list(line_text)], tokenizer=tokenizer, line_sep=None,
                                       seq_len=len(list(line_text)), vocab_dic=word2id, with_real_seq_len=True)
            crf_score = clf.model(torch.LongTensor(sent).to(device), torch.LongTensor(leng).to(device))
            y_score_tag = decode(crf_score.cpu(), torch.LongTensor(sent).gt(0).cpu())[1]

            y_score.append(id2tag(y_score_tag[0][:leng[0]], id2tag_dic))

    def __submit_format(indexs, sent):
        ret = {}
        for start_idx, end_idx in indexs:
            ner_name = sent[start_idx: end_idx + 1]
            if ner_name in ret:
                ret[ner_name].append([start_idx, end_idx])
            else:
                ret[ner_name] = [[start_idx, end_idx]]
        return ret

    def submit(write_path, test_path):
        with open(test_path, "r", encoding='utf-8') as f:
            test_sample = f.readlines()

        with open(write_path, "w", encoding="utf-8") as f:
            line_num = 0
            for i in range(len(y_score)):
                label = {}
                write_line = {"id": line_num}
                tag_entity = parse_entity_from_sequence(y_score[i])
                line_text = json.loads(test_sample[i])["text"]
                for tag in tag_entity:
                    label[tag] = __submit_format(tag_entity[tag], line_text)
                write_line["label"] = label
                f.write(json.dumps(write_line, ensure_ascii=False) + "\n")
                line_num += 1

    submit("./data/cluener/cluener_predict.json", test_path)


if __name__ == "__main__":
    main_entry()
