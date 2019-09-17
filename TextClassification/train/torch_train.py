# -*- coding: utf-8 -*-

from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F


def evaluate(model, data_iter, metric_func):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)

    metric = metric_func(y_true=labels_all, y_pred=predict_all)
    return metric, loss_total / len(data_iter)


def train(train_iter, dev_iter, model, optimizer, num_epochs, metric_func, model_save_path, early_stopping_batch):
    start_time = datetime.now()
    model.train()
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    for epoch in range(num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                y_true = labels.data.cpu()
                y_pred = torch.max(outputs.data, 1)[1].cpu()
                train_metric = metric_func(y_true=y_true, y_pred=y_pred)
                dev_metric, dev_loss = evaluate(model, dev_iter, metric_func)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), model_save_path)
                    last_improve = total_batch

                end_time = datetime.now()
                msg = f'Iter: {total_batch},  Train Loss: {loss.item():.4},  Train metric: {train_metric:.4}, ' \
                      f' Val Loss: {dev_loss:.4},  Val metric: {dev_metric:.4},  Time: {(end_time-start_time).seconds}'
                print(msg)
                model.train()  # modify back train mode
            total_batch += 1
            if total_batch - last_improve > early_stopping_batch:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break


def predict(model, test_iter, model_save_path):
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in test_iter:
            outputs = model(texts)
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)
    return predict_all, labels_all


