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

import time
import numpy as np
import torch
from KDE import put_proto, get_global_proto_kde, get_malicious
from flcore.clients.clientproto import clientProto
from flcore.servers.serverbase import Server
from threading import Thread
from collections import defaultdict, deque


class FedProto(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientProto)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes
        self.global_protos = [None for _ in range(args.num_classes)]
        self.global_protos_var = [None for _ in range(args.num_classes)]
        self.global_protos_skewness = [None for _ in range(args.num_classes)]
        self.max_malicious = 0
        self.kde = args.kde

    def train(self):
        client_Proto_list = {}
        client_Proto_var_list = {}
        client_Proto_skewness_list = {}
        #client_Proto_queues = deque({}, maxlen=2)
        self.max_malicious = 0
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            global_proto_np_list = {}
            global_proto_var_np_list = {}
            global_proto_skewness_np_list = {}
            global_proto_kde = {}
            global_proto_var_kde = {}
            global_proto_skewness_kde = {}
            if i == 0:
                for j, client in enumerate(self.selected_clients):
                    client.id = j

            for client in self.selected_clients:
                # print(j)
                if client.id == 0 or client.id == 1 or client.id == 2:
                    protos_np, protos_var_np, protos_skewness_np = client.train(no_poison=False)
                else:
                    protos_np, protos_var_np, protos_skewness_np = client.train(no_poison=True)
                client_Proto_list[client.id] = protos_np
                client_Proto_var_list[client.id] = protos_var_np
                client_Proto_skewness_list[client.id] = protos_skewness_np
                global_proto_np_list = put_proto(protos_np, global_proto_kde)

            global_proto_kde = get_global_proto_kde(global_proto_np_list, global_proto_kde)
            # global_proto_var_kde = get_global_proto_kde(global_proto_var_np_list, global_proto_var_kde)
            # global_proto_skewness_kde = get_global_proto_kde(global_proto_skewness_np_list, global_proto_skewness_kde)
            for client in self.selected_clients:
                malicious_this_round = get_malicious(client_Proto_list[client.id], global_proto_kde)
                # client.malicious = malicious_this_round
                client.malicious_queue.append(malicious_this_round)
                client.sum_malicious = sum(client.malicious_queue)
                # print("last 5 malicious for " + str(client.id) + ":", client.sum_malicious)
                # print("differ for " + str(client.id) + ":", client.differ_mean)
                # print(str(client.id), client.sum_malicious, client.differ_mean,client.sum_malicious * client.differ_mean)
                print("%-4s: %18s" % (str(client.id), str(client.sum_malicious)))
                # print("differ for " + str(client.id) + ":", client.protos_differ)
            # print("max malicious:", self.max_malicious)
            self.receive_protos(round=i)

            self.global_protos = proto_aggregation(self.uploaded_protos)
            self.global_protos_var = proto_var_aggregation(self.uploaded_protos_var)
            self.global_protos_skewness = proto_var_aggregation(self.uploaded_protos_skewness)
            # print(self.global_protos_skewness)
            self.send_protos()

            self.Budget.append(time.time() - s_t)
            print('-' * 50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()

    def send_protos(self):
        assert (len(self.clients) > 0)
        for client in self.clients:
            start_time = time.time()

            client.set_protos(self.global_protos)
            client.set_protos_var(self.global_protos_var)
            client.set_protos_skewness(self.global_protos_skewness)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_protos(self, round):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_protos = []
        self.uploaded_protos_var = []
        self.uploaded_protos_skewness = []
        malicious_list = []
        for client in self.selected_clients:
            malicious_list.append(client.sum_malicious)
        malicious_list = sorted(malicious_list, reverse=True)
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            if self.kde:
                key_max = malicious_list[2] - 0.0001
                #key_min = malicious_list[-1] + 0.0001
                """
                if client.sum_malicious < key_min and round > 5:
                    # if client.history_Credibility > 3:
                    print("client " + str(client.id) + " is malicious, skip")
                    # client.history_Credibility += 1
                """
                if (client.sum_malicious > key_max and round < 3) or client.history_Credibility >= 2:
                    # if client.history_Credibility > 3:
                    print("client " + str(client.id) + " is malicious, skip")
                    client.history_Credibility += 1
                else:
                    self.uploaded_protos.append(client.protos)
                    self.uploaded_protos_var.append(client.protos_var)
                    self.uploaded_protos_skewness.append(client.protos_skewness)
                    client.history_Credibility -= 1
            else:
                # 不防御
                self.uploaded_protos.append(client.protos)
                self.uploaded_protos_var.append(client.protos_var)
                self.uploaded_protos_skewness.append(client.protos_skewness)

    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))


# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L221
def proto_aggregation(local_protos_list):
    agg_protos_label = defaultdict(list)
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            agg_protos_label[label].append(local_protos[label])
    #print(agg_protos_label)
    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0].data

    return agg_protos_label


def proto_aggregation(local_protos_list):
    agg_protos_label = defaultdict(list)
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            agg_protos_label[label].append(local_protos[label])
    #print(agg_protos_label)
    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0].data

    return agg_protos_label


def proto_var_aggregation(local_protos_list):
    agg_protos_label = defaultdict(list)
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            local_protos[label] = torch.where(torch.isnan(local_protos[label]), torch.full_like(local_protos[label], 0),
                                              local_protos[label])
            agg_protos_label[label].append(local_protos[label])
    for [label, proto_list] in agg_protos_label.items():
        agg_protos_label[label] = proto_list[0].data

    return agg_protos_label


"""
def proto_aggregation_KDE(local_protos_list):
    # print(local_protos_list)
    agg_protos_label = defaultdict(list)
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            agg_protos_label[label].append(local_protos[label])
    # print(agg_protos_label)
    #agg_protos_label = protos_kde(agg_protos_label)

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0].data

    return agg_protos_label
"""


def turn_np(protos):
    protos_np = {}
    for [label, proto_list] in protos.items():
        protos_np[label] = proto_list.clone().detach().cpu().data.numpy()
    return protos_np


def get_cos_similar(v1, v2):
    num = float(np.dot(v1, v2))  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0
