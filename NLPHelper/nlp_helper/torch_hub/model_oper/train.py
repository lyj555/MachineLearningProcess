# -*- coding: utf-8 -*-

import os
import torch

from .model_evaluatioin import evaluate, predict


class SelfModel:
    def __init__(self, model, loss=None, optimizer=None):
        self.loss = loss
        self.optimizer = optimizer
        # train_params
        self.num_epochs_ = None
        self.model_save_path_ = None
        self.feval_, self.is_higher_better_ = None, None
        self._enable_epoch_early_stopping, self._enable_batch_early_stopping = None, None
        self._enable_early_stopping, self._enable_print = None, None
        # self.device_ = self._check_device(device)
        self.device_, self.slice_batch_ = None, None
        self.feval_is_loss_ = False

        self.model = self._check_model(model)  # .to(self.device_)

    def batch_train(self, batch):
        self.model.train()
        batch_data_x, batch_data_y = self.slice_batch_(batch)  # batch shape (x1, x2, ) y
        outputs = self.model(*[i.to(self.device_) for i in batch_data_x])  # position must match
        loss = self.loss_(outputs, batch_data_y.to(self.device_))  # 计算loss

        # self.model.zero_grad()  # 梯度归零
        self.optimizer_.zero_grad()
        loss.backward()  # 梯度反向传播
        self.optimizer_.step()  # 参数更新

    def train(self, train_iter, num_epochs, loss=None, optimizer=None, valid_iter=None, early_stopping_epoch=None,
              model_save_path=None, early_stopping_batch=None, batch_check_frequency=1, device=None,
              feval=None, is_higher_better=True, print_every_epoch=0, print_every_batch=0,
              slice_batch=None, verbose=True):
        """
        模型训练
        :param train_iter: 训练数据迭代器
        :param num_epochs: int, 训练轮数
        :param loss: None or nn.Functional, 损失函数，如果为None，那么__init__方法中必须指定
        :param optimizer: None or nn.optim.Optimizer, 优化算法，如果为None，那么__init__方法中必须指定
        :param valid_iter: None or iterator, 验证集数据，用于控制早停和print
        :param early_stopping_epoch: int, >=0, 连续多少个epoch验证集指标变差后停止训练
        :param model_save_path: str, 如果存在early stop机制，那么必须指定该参数，用于保存最优的模型
        :param early_stopping_batch: int, >=0, 连续多少个batch验证集指标变差后停止训练
        :param batch_check_frequency: int, >=1, 每个多少个batch触发early stop机制或者print机制，
        主要避免batch_size过小的情况下，过多次的计算
        :param device: None or torch.device, 模型训练使用的设备，如果为None使用默认的设备
        :param feval: callable or None, 验证集指标，如果None，默认使用loss作为衡量指标
        :param is_higher_better: bool, 验证集指标是否越高越好，用于early stop的机制
        :param print_every_epoch: int, >=0, 每个多少个epoch进行训练信息打印，如果为0，不打印
        :param print_every_batch: int, >=0, 每个多少个batch进行训练信息打印，如果为0，不打印
        :param slice_batch: None or callable, 对输入batch进行训练数据和标签数据的切分，必须返回两部分，
        默认为None, 表示lambda x: (x[:-1], x[-1], )
        :param verbose: bool, 是否打印一些简单信息，不会对训练时间有额外时间消耗
        :return:
        """
        self._check_params(train_iter, num_epochs, loss, optimizer, valid_iter, early_stopping_epoch, model_save_path,
                           early_stopping_batch, batch_check_frequency, feval, is_higher_better,
                           print_every_epoch, print_every_batch, verbose, device, slice_batch)  # 校验参数
        self.model = self.model.to(self.device_)  # change into cpu or gpu
        counter, metric, stop_train = 0, float("-inf"), False
        global_batch_num, semi_global_batch_num = 0, 0
        for epoch_num in range(1, num_epochs+1):
            if verbose:
                print(f"Epoch Num [{epoch_num}/{num_epochs}]...")
            for batch_num, batch in enumerate(train_iter):
                global_batch_num += 1
                # self.batch_train(batch_x.to(self.device_), batch_y.to(self.device_))
                self.batch_train(batch)
                if global_batch_num % batch_check_frequency == 0:  # 防止计算频繁，引入batch的频率校验
                    semi_global_batch_num += 1
                    # batch 流程
                    if self._enable_batch_early_stopping:
                        valid_metric = self.evaluate(valid_iter, self.feval_, self.feval_is_loss_,
                                                     device=self.device_, slice_batch=self.slice_batch_)
                        counter, metric, stop_train = \
                            self.early_stop_process(counter, early_stopping_batch, metric,
                                                    valid_metric, self.is_higher_better_)
                        if counter == 0:
                            self.save_model(model_save_path)  # save model
                        if stop_train:
                            print(f"trigger early stopping, stop training,"
                                  f" now the epoch number is {epoch_num}/{num_epochs},"
                                  f" the batch number is {batch_num+1}/{len(train_iter)}.")
                            return
                    if print_every_batch > 0 and semi_global_batch_num % print_every_batch == 0:
                        train_metric = self.evaluate(train_iter, self.feval_, self.feval_is_loss_,
                                                     device=self.device_, slice_batch=self.slice_batch_)
                        valid_metric = self.evaluate(valid_iter, self.feval_, self.feval_is_loss_,
                                                     device=self.device_, slice_batch=self.slice_batch_)\
                            if valid_iter is not None else None
                        self._print_metric(epoch_num, num_epochs, batch_num, len(train_iter),
                                           train_metric, valid_metric, print_batch=True)
            # epoch 流程
            if self._enable_epoch_early_stopping:
                valid_metric = self.evaluate(valid_iter, self.feval_, self.feval_is_loss_,
                                             device=self.device_, slice_batch=self.slice_batch_)
                counter, metric, stop_train = \
                    self.early_stop_process(counter, early_stopping_epoch, metric,
                                            valid_metric, self.is_higher_better_)
                if counter == 0:
                    self.save_model(model_save_path)  # save model
                if stop_train:
                    print(f"trigger early stopping, stop training,"
                          f" now the epoch number is {epoch_num}/{num_epochs}.")
                    return
            if print_every_epoch > 0 and epoch_num % print_every_epoch == 0:
                train_metric = self.evaluate(train_iter, self.feval_,
                                             device=self.device_, slice_batch=self.slice_batch_)
                valid_metric = self.evaluate(valid_iter, self.feval_,
                                             device=self.device_, slice_batch=self.slice_batch_) \
                    if valid_iter is not None else None
                self._print_metric(epoch_num, num_epochs, None, None, train_metric, valid_metric, print_batch=False)

    @staticmethod
    def early_stop_process(counter, patience, metric, cur_metric, is_higher_better):
        if not is_higher_better:
            cur_metric = -cur_metric
        if cur_metric > metric:  # 模型效果好
            counter = 0
            return counter, cur_metric, False
        else:
            counter += 1
            return counter, cur_metric, counter >= patience

    def _print_metric(self, epoch_num, num_epochs, batch_num, num_batches,
                      train_metric, valid_metric, print_batch):
        print_name_entity = "loss" if self.feval_is_loss_ else "metric"
        train_metric_desc = f"train {print_name_entity} is {train_metric}"
        valid_metric_desc = f"valid {print_name_entity} is {valid_metric}" if valid_metric is not None else ""
        if print_batch:
            print(f"Epoch Num [{epoch_num}/{num_epochs}], Batch num [{batch_num+1}/{num_batches}]: "
                  f"{train_metric_desc} " + valid_metric_desc)
        else:
            print(f"Epoch Num [{epoch_num}/{num_epochs}]: {train_metric_desc} " + valid_metric_desc)

    def evaluate(self, data, feval=None, count_every_batch=False, y_score_processor=None,
                 slice_batch=None, device=None):
        if slice_batch is None:
            slice_batch = self.slice_batch_
        if device is None:
            device = self.device_
        return evaluate(self.model, data, feval, count_every_batch, y_score_processor, slice_batch, device)

    def predict(self, data, with_label=False, y_score_processor=None, slice_batch=None, device=None):
        if slice_batch is None:
            slice_batch = self.slice_batch_
        if device is None:
            device = self.device_
        return predict(self.model, data, with_label, y_score_processor, slice_batch, device)

    def save_model(self, check_point_path):
        model_info = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer_.state_dict(),
            "loss": self.loss_,
            "epoch": self.num_epochs_}
        torch.save(model_info, check_point_path)

    def load_model(self, model_save_path, load_keys=None, device=None):
        assert os.path.exists(model_save_path), f"input path {model_save_path} not exist."
        assert load_keys is None or isinstance(load_keys, (tuple, list)), "load_keys can be None or list/tuple"
        if device is None:
            device = self.device_

        model_info = torch.load(model_save_path, map_location=device)
        if load_keys is not None:
            for load_key in load_keys:
                assert load_key in model_info, f"input load_keys contain key {load_key}, which don't exist.'"
            model_info = {i: model_info[i] for i in load_keys}

        assert "state_dict" in model_info, "key state_dict must exist!"
        self.model.load_state_dict(model_info["state_dict"])
        # self.model = self.model.to(self.device_)
        if "optimizer" in model_info:
            assert self.optimizer is not None, \
                "optimizer in model info, you must specify torch.optim.Optimizer in __init__ method or" \
                " you can specify param load_keys"
            self.optimizer.load_state_dict(model_info["optimizer"])
        if "loss" in model_info:
            self.loss = model_info["loss"]
        if "epoch" in model_info:
            self.num_epochs_ = model_info["epoch"]

    def _check_params(self, train_iter, num_epochs, loss, optimizer, valid_iter, early_stopping_epoch,
                      model_save_path, early_stopping_batch, batch_check_frequency, feval, is_higher_better,
                      print_every_epoch, print_every_batch, verbose, device, slice_batch):
        self.device_ = self._check_device(device)
        assert isinstance(num_epochs, int) and num_epochs >= 1, f"num_epochs {num_epochs} must be integer(>=1)."
        self.num_epochs_ = num_epochs  # update to self
        self._check_data_iter(train_iter)

        self.loss_, self.optimizer_ = self._check_loss_func(loss), self._check_optimizer(optimizer)  # .to(self.device_)

        if valid_iter is not None:
            self._check_data_iter(valid_iter)
            self._enable_epoch_early_stopping, self._enable_batch_early_stopping = \
                self._check_early_stopping(early_stopping_epoch, early_stopping_batch, batch_check_frequency,
                                           num_epochs, len(train_iter))
        else:
            if (early_stopping_batch is not None and early_stopping_batch > 0) or \
                    (early_stopping_epoch is not None and early_stopping_epoch > 0):
                raise ValueError("when you enable early_stopping param, valid_iter can not be None.")
            self._enable_epoch_early_stopping, self._enable_batch_early_stopping = False, False
        self._enable_early_stopping = self._enable_batch_early_stopping or self._enable_epoch_early_stopping
        self._enable_print = self._check_print_metric(print_every_epoch, print_every_batch)

        if feval is not None:
            self._check_feval(feval)
            self.feval_ = feval
            self.is_higher_better_ = is_higher_better
        else:
            if self._enable_early_stopping or self._enable_print:
                self.feval_ = self.loss_
                self.is_higher_better_ = False
                self.feval_is_loss_ = True

        if self._enable_early_stopping:
            assert model_save_path is not None, \
                f"if enable early stopping, must specify the model save path, which used to save the best model."
            self._check_save_path(model_save_path)
            self.model_save_path_ = model_save_path

        assert isinstance(verbose, bool), "param verbose must be bool type."
        self.slice_batch_ = self._check_slice_batch(slice_batch)

    @staticmethod
    def _check_early_stopping(early_stopping_epoch, early_stopping_batch,
                              batch_check_frequency, num_epoch, num_batch):
        enable_epoch_early_stopping, enable_batch_early_stopping = False, False
        if early_stopping_epoch is None and early_stopping_batch is None:
            return enable_epoch_early_stopping, enable_batch_early_stopping
        elif early_stopping_epoch is not None and early_stopping_batch is None:
            assert isinstance(early_stopping_epoch, int) and (1 <= early_stopping_epoch <= num_epoch), \
                f"if early_stopping_epoch is not None, it must be integer number, and in [1, {num_epoch}]."
            enable_epoch_early_stopping, enable_batch_early_stopping = True, False
        elif early_stopping_epoch is None and early_stopping_batch is not None:
            assert isinstance(early_stopping_batch, int) and (1 <= early_stopping_batch <= num_epoch*num_batch), \
                f"if early_stopping_batch is not None, it must integer number, and in [1, {num_epoch*num_batch}]."
            assert isinstance(batch_check_frequency, int) and (1 <= batch_check_frequency <= num_batch), \
                f"if early_stopping_batch is not None, batch_check_frequency must be integer, and in [1, {num_batch}]."
            enable_epoch_early_stopping, enable_batch_early_stopping = False, True
        else:
            raise ValueError("param enable_epoch_early_stopping and enable_batch_early_stopping "
                             "can not be not-None value at the same time.")
        return enable_epoch_early_stopping, enable_batch_early_stopping

    @staticmethod
    def is_data_iter(data_iter):
        if hasattr(data_iter, "__iter__") and hasattr(data_iter, "__len__"):
            return True
        else:
            return False

    @staticmethod
    def _check_feval(feval):
        assert callable(feval), \
            "input feval can not be None and must be a callable object, like torch.nn.functional's loss"

    @staticmethod
    def is_tensor(data):
        if isinstance(data, torch.Tensor):
            return True
        else:
            return False

    @staticmethod
    def _check_model(model):
        assert isinstance(model, torch.nn.Module), "input model must be torch.nn.Module object"
        # model = copy.deepcopy(model)
        return model

    def _check_data_iter(self, data_iter):
        assert data_iter is not None, "data iter can not be None object."
        assert self.is_data_iter(data_iter), \
            f"data iter must have attributes `__next__`, `__iter__`, `__len__`."

    def _check_loss_func(self, loss):
        if loss is None and self.loss is None:
            raise ValueError("The loss can not be None, you can set in __init__ method or train method!")
        elif loss is None and self.loss is not None:
            assert callable(self.loss), "input loss must be a callable object, like torch.nn.functional's loss"
            return self.loss
        elif loss is not None and self.loss is None:
            assert callable(loss), "input loss must be a callable object, like torch.nn.functional's loss"
            return loss
        else:
            print("Both the __init__ and train method specify the loss function, the train method loss will be used!")
            assert callable(loss), "input loss must be a callable object, like torch.nn.functional's loss"
            return loss

    def _check_optimizer(self, optimizer):
        if optimizer is None and self.optimizer is None:
            raise ValueError("The optimizer can not be None, you can set in __init__ method or train method!")
        elif optimizer is None and self.optimizer is not None:
            assert isinstance(self.optimizer, torch.optim.Optimizer), \
                "input optimizer must be a torch.optim.Optimizer object."
            return self.optimizer
        elif optimizer is not None and self.optimizer is None:
            assert isinstance(optimizer, torch.optim.Optimizer), \
                "input optimizer must be a torch.optim.Optimizer object."
            return optimizer
        else:
            print("Both the __init__ and train method specify the optimizer, the train method optimizer will be used!")
            assert isinstance(optimizer, torch.optim.Optimizer), \
                "input optimizer must be a torch.optim.Optimizer object."
            return optimizer

    @staticmethod
    def _check_print_metric(print_every_epoch, print_every_batch):
        assert isinstance(print_every_epoch, int) and print_every_epoch >= 0, \
            f"param print_every_epoch must greater than or equal to 0"
        assert isinstance(print_every_batch, int) and print_every_batch >= 0, \
            f"param print_every_batch must greater than or equal to 0"
        if print_every_batch > 0 or print_every_epoch > 0:
            enable_print = True
        else:
            enable_print = False
        return enable_print

    @staticmethod
    def _check_save_path(path):
        base_dir = os.path.dirname(path)
        assert os.path.exists(base_dir) and os.path.isdir(base_dir), \
            f"parent directory {base_dir} must exist and is a valid directory."
        if os.path.exists(path):
            assert os.path.isfile(path), "if input path exist, it must be a valid file."

    @staticmethod
    def _check_device(device):
        assert device is None or isinstance(device, bool) or isinstance(device, torch.device), \
            "param device must be None or bool type or torch.device type."
        if device is None:   # use default device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, bool):
            if device:
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                else:
                    print("param device set True, but cuda is not available, so change into cpu model.")
                    device = torch.device("cpu")
            else:
                device = torch.device("cpu")
        else:
            pass
        return device

    @staticmethod
    def _check_slice_batch(slice_batch):
        assert slice_batch is None or callable(slice_batch), "param slice_batch must be None or callable object."
        if slice_batch is None:
            return lambda x: (x[:-1], x[-1], )
        else:
            return slice_batch


# if __name__ == "__main__":
#     import numpy as np
#     import torch
#     from torch.utils.data import TensorDataset, DataLoader
#
#     sample_num, feature_num = 100, 3
#
#     x = 10 * np.random.rand(sample_num, feature_num)
#     y = np.random.choice([1, 0], size=(sample_num, 1), replace=True, p=[0.4, 0.6]).reshape(-1, 1)
#
#     valid_x = np.random.rand(20, feature_num)
#     valid_y = np.random.choice([1, 0], size=(20, 1), replace=True, p=[0.4, 0.6]).reshape(-1, 1)
#
#     train_iter = DataLoader(TensorDataset(torch.Tensor(x), torch.Tensor(y)), 3, shuffle=False)
#     valid_iter = DataLoader(TensorDataset(torch.Tensor(valid_x), torch.Tensor(valid_y)), 3)
#
#
#     class LogisticRegression(torch.nn.Module):
#         def __init__(self, input_dim):
#             super(LogisticRegression, self).__init__()
#             self.linear = torch.nn.Linear(input_dim, 1)  # input and output is 1 dimension
#             self.sigmoid = torch.nn.Sigmoid()
#
#         def forward(self, x):
#             pred = self.sigmoid(self.linear(x))
#             return pred
#
#
#     model = LogisticRegression(input_dim=x.shape[1])  # 初始化模型
#     criterion = torch.nn.BCELoss(reduction='mean')  # 定义损失（返回的损失求均值）
#     optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)  # 定义优化算法
#
#     # 测试基本功能
#     clf = SelfModel(model=model)
#     clf.train(train_iter, 10, loss=criterion, optimizer=optimizer, verbose=True)
#
#     print(clf.evaluate(train_iter, criterion))
#     print(clf.predict(train_iter, with_label=True))
#
#     # 测试epoch打印
#     clf = SelfModel(model=model)
#     clf.train(train_iter, 10, loss=criterion, optimizer=optimizer, print_every_epoch=2)
#
#     print(clf.evaluate(train_iter, criterion))
#     print(clf.predict(train_iter, with_label=True))
#
#     clf = SelfModel(model=model)
#     clf.train(train_iter, 10, loss=criterion, optimizer=optimizer, print_every_epoch=2, valid_iter=train_iter)
#
#     print(clf.evaluate(train_iter, criterion))
#     print(clf.predict(train_iter, with_label=True))
#
#     # 测试batch打印
#     clf = SelfModel(model=model)
#     clf.train(train_iter, 10, loss=criterion, optimizer=optimizer, print_every_batch=2)
#
#     clf = SelfModel(model=model)
#     clf.train(train_iter, 10, loss=criterion, optimizer=optimizer, print_every_batch=2, valid_iter=train_iter)
#
#     clf = SelfModel(model=model)
#     clf.train(train_iter, 10, loss=criterion, optimizer=optimizer, print_every_batch=2,
#               batch_check_frequency=3, valid_iter=train_iter)
#
#     # 测试early_stopping_epoch
#     clf = SelfModel(model=model)
#     clf.train(train_iter, 10, loss=criterion, optimizer=optimizer, print_every_batch=2, early_stopping_epoch=3)  # error
#
#     clf = SelfModel(model=model)
#     clf.train(train_iter, 10, loss=criterion, optimizer=optimizer, print_every_batch=0,
#               early_stopping_epoch=3, valid_iter=valid_iter, model_save_path="./ta.pkl")
#
#     clf = SelfModel(model=model)
#     clf.train(train_iter, 10, loss=criterion, optimizer=optimizer, print_every_batch=0,
#               early_stopping_batch=4, valid_iter=valid_iter, model_save_path="./ta.pkl")
#
#     clf = SelfModel(model=model)
#     clf.train(train_iter, 10, loss=criterion, optimizer=optimizer, print_every_batch=2,
#               early_stopping_epoch=3, early_stopping_batch=2, valid_iter=train_iter, model_save_path="./ta.pkl")  # error
#
#     # 加载模型
#     clf = SelfModel(model=model, optimizer=optimizer)
#     clf.load_model(model_save_path="./ta.pkl")
#
#     clf = SelfModel(model=model)
#     clf.load_model(model_save_path="./ta.pkl")  # error
#
#     clf = SelfModel(model=model)
#     clf.load_model(model_save_path="./ta.pkl", load_keys=["state_dict"])
#
#     # 模型复现
#     # 未使用随机种子
#     model = LogisticRegression(input_dim=x.shape[1])  # 初始化模型
#
#     clf = SelfModel(model=model)
#     clf.train(train_iter, num_epochs=100, loss=criterion, optimizer=optimizer, print_every_batch=5)
#     y1 = clf.predict(valid_iter, with_label=True)
#
#     model = LogisticRegression(input_dim=x.shape[1])  # 初始化模型
#     clf = SelfModel(model=model)
#     clf.train(train_iter, num_epochs=100, loss=criterion, optimizer=optimizer)
#     y2 = clf.predict(valid_iter, with_label=True)
#     print(all(y1[0] == y2[0]))
#
#     # 使用随机种子
#     seed = 789
#     seed_everything(seed, use_np=False, use_gpu=False)
#     model = LogisticRegression(input_dim=x.shape[1])  # 初始化模型
#     clf = SelfModel(model=model)
#     clf.train(train_iter, num_epochs=100, loss=criterion, optimizer=optimizer)
#     y1 = clf.predict(valid_iter, with_label=True)
#
#     seed_everything(seed, use_np=False, use_gpu=False)
#     model = LogisticRegression(input_dim=x.shape[1])  # 初始化模型
#     clf = SelfModel(model=model)
#     clf.train(train_iter, num_epochs=100, loss=criterion, optimizer=optimizer)
#     y2 = clf.predict(valid_iter, with_label=True)
#     print(all(y1[0] == y2[0]))
#
#     # 测试cpu 和 gpu
#     model = LogisticRegression(input_dim=x.shape[1])  # 初始化模型
#     clf = SelfModel(model=model)
#     clf.train(train_iter, num_epochs=100, loss=criterion, optimizer=optimizer)
#
#     model = LogisticRegression(input_dim=x.shape[1])  # 初始化模型
#     clf = SelfModel(model=model)
#     clf.train(train_iter, num_epochs=100, loss=criterion, optimizer=optimizer, device=torch.device("cpu"))
#
#     model = LogisticRegression(input_dim=x.shape[1])  # 初始化模型
#     clf = SelfModel(model=model)
#     clf.train(train_iter, num_epochs=100, loss=criterion, optimizer=optimizer, device=torch.device("gpu"))
#
#     for epoch in range(3):
#         for x_train, y_train in train_iter:
#             # forward
#             out = model(x_train)
#             loss = criterion(out, y_train)
#             # backward
#             optimizer.zero_grad()  # 梯度归零
#             loss.backward()  # 梯度反向传播
#             optimizer.step()  # 参数更新
#         print(epoch, loss)
#
#     # evaluate和predict
#     model = LogisticRegression(input_dim=x.shape[1])  # 初始化模型
#     clf = SelfModel(model=model)
#     clf.train(train_iter, num_epochs=100, loss=criterion, optimizer=optimizer)
#
#     y_score, y_true = evaluate(model=clf.model, data=valid_iter)  # 无feval，返回两部分，分别为预测值和标签
#     assert len(y_score) == len(y_true)
#
#     y_score, y_true = evaluate(model=clf.model, data=valid_iter)  # 无feval，返回两部分，分别为预测值和标签
#     assert len(y_score) == len(y_true)
#     evaluate(model=clf.model, data=valid_iter, feval=criterion)  # 只有一个值
#
#     evaluate(model=clf.model, data=valid_iter, feval=criterion, count_every_batch=True)  # 只有一个值
#
#
