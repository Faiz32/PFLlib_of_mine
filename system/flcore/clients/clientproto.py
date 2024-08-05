# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from collections import defaultdict


def compute_variance(prototypes):
    """
    计算每个类别原型的方差。
    """
    variances = {}
    for label, reps in prototypes.items():
        if len(reps) > 1:
            stacked_reps = torch.stack(reps, dim=0)
            variances[label] = torch.var(stacked_reps, dim=0)
        else:
            variances[label] = torch.zeros_like(reps[0])
    return variances


def compute_skewness(prototypes):
    """
    计算每个类别原型的方差。
    """
    skewness_dict = {}
    for label, reps in prototypes.items():
        if len(reps) > 1:
            skewness_dict[label] = skewness(reps)
        else:
            skewness_dict[label] = torch.zeros_like(reps[0])
    return skewness_dict


def skewness(prototypes):
    x = torch.stack(prototypes, dim=0)
    mean = torch.mean(x, dim=0)
    std = torch.std(x, dim=0)
    return torch.mean((torch.div((x - mean), std)).pow(3), dim=0)


class clientProto(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.protos = None
        self.protos_var = None
        self.protos_skewness = None
        self.global_protos = None
        self.global_protos_var = None
        self.global_protos_skewness = None
        self.loss_mse = nn.MSELoss()

        self.lamda = args.lamda
        self.beta = args.beta
        # self.beta = 3.0
        self.gamma = 0.0

    def train(self):
        """
        训练模型的过程。
        此函数不接受参数，也不返回值。
        """

        # 加载训练数据
        trainloader = self.load_train_data()
        start_time = time.time()

        # 将模型设置为训练模式
        self.model.train()

        # 根据是否进行慢速训练，随机确定本地训练轮数
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        protos = defaultdict(list)
        protos_var = defaultdict(list)
        protos_skewness = defaultdict(list)
        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                # 将数据移动到指定设备上
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                # 如果设置为慢速训练，则随机睡眠以模拟延迟
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                # 前向传播
                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)

                # 如果定义了全局原型，则在损失函数中加入对原型的更新
                if self.global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(self.global_protos[y_c]) != type([]):
                            proto_new[i, :] = self.global_protos[y_c].data
                    loss += self.loss_mse(proto_new, rep) * self.lamda

                # 记录每个类别对应的特征表示
                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

                for protos_key, protos_value in protos.items():
                    protos_var[protos_key].append(torch.var(torch.stack(protos_value, dim=0), dim=0))
                    protos_skewness[protos_key].append(skewness(protos_value))

                if self.global_protos_var is not None:
                    rep_var = compute_variance(protos)
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if self.global_protos_var[y_c] is not None:
                            proto_var_new = self.global_protos_var[y_c].data
                            loss += self.loss_mse(rep_var[y_c], proto_var_new) * self.beta

                if self.global_protos_skewness is not None:
                    rep_skewness = compute_skewness(protos)
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if self.global_protos_skewness[y_c] is not None:
                            proto_skewness_new = self.global_protos_skewness[y_c].data
                            loss += self.loss_mse(rep_skewness[y_c], proto_skewness_new) * self.gamma

                # 反向传播和参数更新
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # 更新全局原型并应用学习率衰减
        self.protos = agg_func(protos)
        self.protos_var = agg_func(protos_var)
        self.protos_skewness = agg_func(protos_skewness)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        # 更新训练时间统计
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_protos(self, global_protos):
        self.global_protos = global_protos

    def set_protos_var(self, global_protos_var):
        self.global_protos_var = global_protos_var

    def set_protos_skewness(self, global_protos_skewness):
        self.global_protos_skewness = global_protos_skewness

    def collect_protos(self):
        trainloader = self.load_train_data()
        self.model.eval()
        protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = self.model.base(x)

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

        self.protos = agg_func(protos)

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0

        if self.global_protos is not None:
            with torch.no_grad():
                for x, y in testloaderfull:
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    rep = self.model.base(x)

                    output = float('inf') * torch.ones(y.shape[0], self.num_classes).to(self.device)
                    for i, r in enumerate(rep):
                        for j, pro in self.global_protos.items():
                            if type(pro) != type([]):
                                output[i, j] = self.loss_mse(r, pro)

                    test_acc += (torch.sum(torch.argmin(output, dim=1) == y)).item()
                    test_num += y.shape[0]

            return test_acc, test_num, 0
        else:
            return 0, 1e-5, 0

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)

                if self.global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(self.global_protos[y_c]) != type([]):
                            proto_new[i, :] = self.global_protos[y_c].data
                    loss += self.loss_mse(proto_new, rep) * self.lamda
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num


# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L205
def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos
