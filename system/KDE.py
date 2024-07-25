import copy
import math

import numpy as np
from scipy.stats import gaussian_kde
from collections import defaultdict


def kde(proto):
    has_nan = np.any(np.isnan(proto))
    if has_nan:
        # proto = np.delete(proto, np.where(np.isnan(proto)))
        proto = np.where(np.any(np.isnan(proto)), 0, proto)
    lambda_val = 1e-6
    if proto.shape[1] <= 2:
        proto_A = proto + lambda_val * np.random.rand(np.shape(proto)[0], np.shape(proto)[1])
        proto_B = proto + lambda_val * np.random.rand(np.shape(proto)[0], np.shape(proto)[1])
        # proto = np.append(proto, proto_A, axis=1)
        proto = np.append(proto_A, proto_B, axis=1)
    # print(len(proto))
    proto_T = proto.T
    # print(proto_T)
    kde_list = []
    for i in range(len(proto_T)):
        kde_value = gaussian_kde(proto_T[i])
        kde_list.append(kde_value)
    return kde_list


def put_proto(protos_np, global_proto_np_list):
    for label, value in protos_np.items():
        if label not in global_proto_np_list:
            global_proto_np_list[label] = value
        else:
            global_proto_np_list[label] = np.append(global_proto_np_list[label], value, axis=0)
    return global_proto_np_list


def get_global_proto_kde(global_proto_np_list, global_proto_kde):
    for label, value in global_proto_np_list.items():
        x = np.vstack(value)
        global_proto_kde[label] = kde(x)
    return global_proto_kde


def get_malicious(client_Proto, global_proto_kde):
    malicious_list = []
    for label, value in client_Proto.items():
        for i in range(len(global_proto_kde[label])):
            evaluate_value = global_proto_kde[label][i].evaluate(value[i])
            malicious = 1 / evaluate_value
            malicious_list.append(malicious)
    malicious_mean = np.mean(malicious_list)
    """
        # print(evaluate_value)
    if evaluate_value > 0.5:
        return 1
    else:
        return 0
    """

    return math.log(malicious_mean)
