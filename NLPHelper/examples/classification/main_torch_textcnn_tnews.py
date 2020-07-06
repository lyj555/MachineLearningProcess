# -*- coding: utf-8 -*-

import os
import json
from datetime import datetime
import numpy as np
import torch
from sklearn.metrics import accuracy_score

from nlp_helper.torch_hub.data_loader import self_iterator, torch_iterator
from nlp_helper.torch_hub.models import TextRNN, TextCNN
from nlp_helper.torch_hub.model_oper import init_network, SelfModel, seed_everything, evaluate

from nlp_helper.utils import build_vocab_by_raw_file, split_data_with_index, sample_data_by_label
from nlp_helper.pre_process import content_to_id


def get_label_map(label_path):
    with open(label_path, "r") as f:
        label_dic = {}
        label_ind = 0
        for line in f:
            line = line.strip("\n")
            line_json = json.loads(line)
            label_dic[line_json["label"]] = {"label_index": label_ind,
                                             "label_desc": line_json["label_desc"]}
            label_ind += 1
        return label_dic


def transform_data(data_path, label_dic, out_path):
    f_in = open(data_path, "r")
    f_out = open(out_path, "w")

    try:
        for line in f_in:
            line = line.strip("\n")
            line_json = json.loads(line)

            if "label" in line_json:
                new_line = f"{line_json['keywords']};{line_json['sentence']}\t{label_dic[line_json['label']]['label_index']}\n"
            else:
                new_line = f"{line_json['keywords']};{line_json['sentence']}\n"
            f_out.write(new_line)
    finally:
        f_in.close()
        f_out.close()


def main_entry(save_dir):
    # seed_everything(987, use_np=True, use_cpu=True, use_gpu=False)

    # [0]. 转换数据格式, 形如 sentences+label
    train_pri_path = "./data/tnews/train.json"
    train_path = "./data/tnews/train_trans.txt"

    valid_pri_path = "./data/tnews/dev.json"
    valid_path = "./data/tnews/dev_trans.txt"

    test_pri_path = "./data/tnews/test.json"
    test_path = "./data/tnews/test_trans.txt"

    label_path = "./data/tnews/labels.json"
    label_dic = get_label_map(label_path)
    transform_data(train_pri_path, label_dic, train_path)
    transform_data(valid_pri_path, label_dic, valid_path)
    transform_data(test_pri_path, label_dic, test_path)

    # [1]. 创建词汇表字典
    # [1.1]. 无词汇表，从指定文件创建并保存
    vocab_file_path = train_path
    save_path = os.path.join(save_dir, "train_vocab.pkl")
    tokenizer = "char"
    line_sep = "\t"

    vocab_dic = build_vocab_by_raw_file(vocab_file_path, line_sep=line_sep,
                                        tokenizer=tokenizer, word_dic_save_path=save_path)
    # [1.2]. 有词汇表，从指定文件创建
    # [1.3]. 有词汇表，手动从pickle文件中加载
    # [1.4]. 有词汇表，基于此进行更新

    # [2]. 文本转换为id
    # train_path = "./data/THUCNews/train.txt"
    # valid_path = "./data/THUCNews/dev.txt"
    # test_path = "./data/THUCNews/test.txt"
    seq_len = 100

    train_x, train_y = content_to_id(train_path, tokenizer=tokenizer, seq_len=seq_len,
                                     vocab_dic=vocab_dic, line_sep=line_sep)
    print(f"train_x sample number is {len(train_x)}, label sample number is {len(train_y)}")

    valid_x, valid_y = content_to_id(valid_path, tokenizer=tokenizer, seq_len=seq_len,
                                     vocab_dic=vocab_dic, line_sep=line_sep)
    print(f"valid_x sample number is {len(valid_x)}, label sample number is {len(valid_y)}")

    # test_x, test_y = content_to_id(test_path, tokenizer=tokenizer, seq_len=seq_len,
    #                                vocab_dic=vocab_dic, line_sep=line_sep)
    # print(f"content sample number is {len(test_x)}, label sample number is {len(test_y)}")
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
        train_x, train_y = train_x[:small_sample_num], train_y[:small_sample_num]

    train_iter = torch_iterator(batch_data=(train_x, train_y,), batch_size=batch_size)
    valid_iter = torch_iterator(batch_data=(valid_x, valid_y,), batch_size=batch_size)
    # test_iter = torch_iterator(batch_data=(test_x, test_y), batch_size=batch_size)

    # [6]. 初始化模型
    seed_everything(1024, use_np=True, use_cpu=True, use_gpu=True)

    # model = TextRNN(vocab_size=len(vocab_dic), embedding_dim=8, hidden_size=20,
    #                 num_layers=2, num_classes=10, dropout=0.5)
    model = TextCNN(num_filters=128, filter_sizes=(2, 3, 4), num_classes=len(label_dic), vocab_size=len(vocab_dic),
                    embedding_dim=300, dropout=0.5)
    init_network(model)
    print(model)

    # [7]. 模型训练
    num_epochs = 6
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
    # Epoch Num [6/6], Batch num [395/417]: train loss is 0.5875816802612598 valid loss is 1.1788143689119364
    # Epoch Num [6/6], Batch num [415/417]: train loss is 0.5919032108297737 valid loss is 1.1893426436412184
    # train cost 2202 seconds

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
    # y_score, y_true = evaluate(clf.model, test_iter, y_score_processor=get_max_prob_index)
    # test_acc = accuracy_score(y_true, y_score)
    print(f"train accuracy is {train_acc}, valid accuracy is {valid_acc}.")
    # train accuracy is 0.8219827586206897, valid accuracy is 0.6129.

    # [10]. 对测试集进行预测, 构造线上cluemark提交格式, 提交到线上查看效果
    inverse_label_dic = {}
    for key, val in label_dic.items():
        inverse_label_dic[val["label_index"]] = {"label": key, "label_desc": val["label_desc"]}

    f_out = open("./data/tnews/tnews_predict.json", "w", encoding="utf-8")

    with open(test_path, "r") as f:
        line_num = 0
        for line in f:
            line_json = {"id": line_num}
            line = line.strip("\n")
            line_ids = content_to_id([line], tokenizer=tokenizer, seq_len=seq_len, vocab_dic=vocab_dic)
            line_pred = clf.model(torch.LongTensor(line_ids).to(device))  # 返回预测每个类别的预测
            line_pred_ind = torch.max(line_pred, 1)[1].item()  # 获取最大概率对应的index
            line_json.update(inverse_label_dic[line_pred_ind])  # 构造线上格式
            f_out.write(f"{json.dumps(line_json, ensure_ascii=False)}\n")  # 写入文件中
            line_num += 1
    f_out.close()


if __name__ == "__main__":
    print(os.listdir("./"))
    save_dir = "/data/tnews"
    main_entry(save_dir)
