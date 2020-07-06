# -*- coding: utf-8 -*-

import os
from datetime import datetime
import numpy as np
import torch
from sklearn.metrics import accuracy_score

from nlp_helper.torch_hub.data_loader import self_iterator, torch_iterator
from nlp_helper.torch_hub.models import TextRNN, TextCNN
from nlp_helper.torch_hub.model_oper import init_network, SelfModel, seed_everything, evaluate

from nlp_helper.utils import build_vocab_by_raw_file, split_data_with_index, sample_data_by_label
from nlp_helper.pre_process import content_to_id


def main_entry(save_dir):
    # seed_everything(987, use_np=True, use_cpu=True, use_gpu=False)
    # [1]. 创建词汇表字典
    # [1.1]. 无词汇表，从指定文件创建并保存
    vocab_file_path = "./data/THUCNews/train.txt"
    save_path = os.path.join(save_dir, "train_vocab.pkl")
    tokenizer = "char"
    line_sep = "\t"

    vocab_dic = build_vocab_by_raw_file(vocab_file_path, line_sep=line_sep,
                                        tokenizer=tokenizer, word_dic_save_path=save_path)
    # [1.2]. 有词汇表，从指定文件创建
    # [1.3]. 有词汇表，手动从pickle文件中加载
    # [1.4]. 有词汇表，基于此进行更新

    # [2]. 文本转换为id
    train_path = "./data/THUCNews/train.txt"
    valid_path = "./data/THUCNews/dev.txt"
    test_path = "./data/THUCNews/test.txt"
    seq_len = 32

    train_x, train_y = content_to_id(train_path, tokenizer=tokenizer, seq_len=seq_len,
                                     vocab_dic=vocab_dic, line_sep=line_sep)
    print(f"train_x sample number is {len(train_x)}, label sample number is {len(train_y)}")

    valid_x, valid_y = content_to_id(valid_path, tokenizer=tokenizer, seq_len=seq_len,
                                     vocab_dic=vocab_dic, line_sep=line_sep)
    print(f"valid_x sample number is {len(valid_x)}, label sample number is {len(valid_y)}")

    test_x, test_y = content_to_id(test_path, tokenizer=tokenizer, seq_len=seq_len,
                                   vocab_dic=vocab_dic, line_sep=line_sep)
    print(f"content sample number is {len(test_x)}, label sample number is {len(test_y)}")
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
    small_sample_test = True
    small_sample_num = 1000
    if small_sample_test:
        train_x, train_y = train_x[:small_sample_num], train_y[:small_sample_num]

    train_iter = torch_iterator(batch_data=(train_x, train_y,), batch_size=batch_size)
    valid_iter = torch_iterator(batch_data=(valid_x, valid_y,), batch_size=batch_size)
    test_iter = torch_iterator(batch_data=(test_x, test_y), batch_size=batch_size)

    # [6]. 初始化模型
    seed_everything(1024, use_np=True, use_cpu=True, use_gpu=True)

    # model = TextRNN(vocab_size=len(vocab_dic), embedding_dim=8, hidden_size=20,
    #                 num_layers=2, num_classes=10, dropout=0.5)
    model = TextCNN(num_filters=128, filter_sizes=(2, 3, 4), num_classes=10, vocab_size=len(vocab_dic),
                    embedding_dim=300, dropout=0.5)
    init_network(model)
    print(model)

    # [7]. 模型训练
    num_epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 1e-3
    model_save_path = os.path.join(save_dir, "text_cnn_model.pt")  # "./data/THUCNews/text_cnn_model.pt"
    print("now the device is ", device)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    clf = SelfModel(model=model)
    t1 = datetime.now()
    clf.train(train_iter, num_epochs, loss=loss, optimizer=optimizer, valid_iter=valid_iter,
              early_stopping_batch=100, batch_check_frequency=2,
              print_every_batch=10, model_save_path=model_save_path, device=device)
    t2 = datetime.now()
    print(f"train cost {(t2-t1).seconds} seconds")

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


if __name__ == "__main__":
    # print(os.listdir("./"))
    save_dir = "./data/THUCNews"
    main_entry(save_dir)
