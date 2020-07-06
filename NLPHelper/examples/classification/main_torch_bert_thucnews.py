# -*- coding: utf-8 -*-

import os
from datetime import datetime
import numpy as np
import torch
from sklearn.metrics import accuracy_score

from nlp_helper.torch_hub.data_loader import self_iterator, torch_iterator
from nlp_helper.torch_hub.models.bert import Bert
from nlp_helper.torch_hub.models.pytorch_pretrained import BertTokenizer, BertAdam
from nlp_helper.torch_hub.model_oper import init_network, SelfModel, seed_everything, evaluate

from nlp_helper.utils import build_vocab_by_raw_file, split_data_with_index, sample_data_by_label
from nlp_helper.pre_process import content_to_id


START_TOKEN = "[CLS]"
END_TOKEN = "[SEP]"


# def make_sequence_mask(real_len, seq_len):
#     assert real_len <= seq_len, f"real_len: {real_len} > seq_len: {seq_len}"
#     return [1]*real_len+[0]*(seq_len-real_len)


def construct_data(path, vocab_dic, tokenizer, seq_len, line_sep):
    # cls_index = vocab_dic.get("[CLS]")  # 首位置添加[CLS]
    x, y, lengths = content_to_id(path, tokenizer=tokenizer, seq_len=seq_len,
                                  vocab_dic=vocab_dic, line_sep=line_sep, with_real_seq_len=True)
    # x = np.insert(x, 0, cls_index, axis=1)
    # mask = np.array([make_sequence_mask(real_len=i + 1, seq_len=seq_len + 1) for i in lengths])  # 多加了一个cls_index
    mask = (x > 0).astype(int)
    print(f"x sample number is {len(x)}, label sample number is {len(y)}")
    return x, y, mask


def get_bert_optimizer(named_parameters, learning_rate, t_total):
    param_optimizer = list(named_parameters)
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=learning_rate,
                         warmup=0.05,
                         t_total=t_total)
    return optimizer


def main_entry(save_dir):
    bert_pretrain_path = "./data/bert_pretrain_file/torch"
    # seed_everything(987, use_np=True, use_cpu=True, use_gpu=False)
    # [1]. 创建词汇表字典
    # [1.1]. 无词汇表，从指定文件创建并保存
    # vocab_dic = build_vocab_by_raw_file(vocab_file_path, line_sep=line_sep,
    #                                     tokenizer=tokenizer, word_dic_save_path=save_path)
    # [1.2]. 有词汇表，从指定文件创建
    # [1.3]. 有词汇表，手动从pickle文件中加载
    # [1.4]. 有词汇表，基于此进行更新
    bert_tokenizer = BertTokenizer.from_pretrained(bert_pretrain_path)
    vocab_dic = bert_tokenizer.vocab

    # [2]. 文本转换为id
    train_path = "./data/THUCNews/train.txt"
    valid_path = "./data/THUCNews/dev.txt"
    test_path = "./data/THUCNews/test.txt"

    tokenizer = lambda x: [START_TOKEN] + bert_tokenizer.tokenize(x) + [END_TOKEN]
    seq_len = 32
    line_sep = "\t"

    train_x, train_y, train_mask = construct_data(train_path, vocab_dic, tokenizer, seq_len, line_sep)
    valid_x, valid_y, valid_mask = construct_data(valid_path, vocab_dic, tokenizer, seq_len, line_sep)
    test_x, test_y, test_mask = construct_data(test_path, vocab_dic, tokenizer, seq_len, line_sep)

    # [3]. 切分数据为三部分（训练、验证和测试集），（随机切分或者标签比例切分）
    # 当然如果第二步已经切分，则此部分可以忽略
    # train_ind, valid_ind, test_ind = split_data_with_index(indexes=len(content), split_ratios=(0.7, 0.1, 0.2))
    # train_x, train_y = np.array(content)[train_ind], np.array(label)[train_ind]
    # valid_x, valid_y = np.array(content)[valid_ind], np.array(label)[valid_ind]
    # test_x, test_y = content[test_ind], label[test_ind]
    # 也有可能[2]和[3]颠倒，即我首先读取数据，可以通过pandas等等方式，先处理数据，
    # 然后通过切分策略，切分数据，此时数据已经划分为三部分或者两部分，然后再走[2]，将这两部分逻辑分别写例子

    # [4]. 数据策略，比如按类别做上采样，下采样；
    # 此时数据已经为numpy格式
    # for i in np.unique(train_y):
    #     print(f"label {i} number is {sum(train_y == i)}")
    # sample_ind = sample_data_by_label(train_y, sampler={"1": 10, "2": 20})
    # train_x, train_y = train_x[sample_ind], train_y[sample_ind]
    # for i in np.unique(train_y):
    #     print(f"label {i} number is {sum(train_y == i)}")

    # [5]. 构建Iterator
    # train_iter = self_iterator(batch_data=(train_x, train_y, ), batch_size=4, )
    # valid_iter = self_iterator(batch_data=(valid_x, valid_y, ), batch_size=4)
    # test_iter = self_iterator(batch_data=(test_x, test_y), batch_size=4)
    batch_size = 128
    small_sample_test = False
    small_sample_num = 10000
    if small_sample_test:
        train_x, train_mask, train_y = train_x[:small_sample_num], train_mask[:small_sample_num], train_y[:small_sample_num]

    train_iter = torch_iterator(batch_data=(train_x, train_mask, train_y,), batch_size=batch_size)
    valid_iter = torch_iterator(batch_data=(valid_x, valid_mask, valid_y,), batch_size=batch_size)
    test_iter = torch_iterator(batch_data=(test_x, test_mask, test_y,), batch_size=batch_size)

    # [6]. 初始化模型
    seed_everything(1024, use_np=True, use_cpu=True, use_gpu=True)

    model = Bert(bert_pretrain_path, hidden_size=768, num_classes=10)
    # init_network(model)
    print(model)

    # [7]. 模型训练
    num_epochs = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save_path = os.path.join(save_dir, "text_cnn_model.pt")  # "./data/THUCNews/text_cnn_model.pt"
    print("now the device is ", device)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = get_bert_optimizer(model.named_parameters(), learning_rate=5e-5, t_total=len(train_iter)*num_epochs)

    clf = SelfModel(model=model)
    t1 = datetime.now()
    clf.train(train_iter, num_epochs, loss=loss, optimizer=optimizer, valid_iter=valid_iter,
              early_stopping_epoch=1, batch_check_frequency=2,
              print_every_batch=20, model_save_path=model_save_path, device=device)
    t2 = datetime.now()
    print(f"train cost {(t2-t1).seconds} seconds")
    # Epoch Num [1/1], Batch num [1360/1407]: train loss is 0.11314889871235341 valid loss is 0.17904165726673754
    # Epoch Num [1/1], Batch num [1400/1407]: train loss is 0.11242205112134639 valid loss is 0.17842706849303427
    # train cost 24566 seconds

    # [8]. 模型预测
    # pred = clf.predict(data=train_iter, do_func=lambda x: x[0])

    # [9]. 查看效果
    def get_max_prob_index(pred):
        return torch.max(pred, 1)[1]

    # pred = torch.nn.functional.softmax(pred, dim=1).cpu().numpy()

    y_score, y_true = evaluate(clf.model, train_iter, y_score_processor=get_max_prob_index)
    train_acc = accuracy_score(y_true, y_score)
    y_score, y_true = evaluate(clf.model, valid_iter, y_score_processor=get_max_prob_index)
    valid_acc = accuracy_score(y_true, y_score)
    y_score, y_true = evaluate(clf.model, test_iter, y_score_processor=get_max_prob_index)
    test_acc = accuracy_score(y_true, y_score)
    print(f"train accuracy is {train_acc}, valid accuracy is {valid_acc}, test accuracy is {test_acc}.")
    # train accuracy is 0.9634777777777778, valid accuracy is 0.9406, test accuracy is 0.9461.


if __name__ == "__main__":
    save_dir = "./data/THUCNews"
    main_entry(save_dir)
