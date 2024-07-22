import copy
import numpy as np
from scipy.stats import gaussian_kde
from collections import defaultdict


def kde(proto):
    proto = np.append(proto, proto)
    kde_list = gaussian_kde(proto.T)
    return kde_list


def put_proto(protos_np, global_proto_np_list):
    for label, value in protos_np.items():
        if label not in global_proto_np_list:
            global_proto_np_list[label] = global_proto_np_list[label]
        else:
            global_proto_np_list[label] = np.append(global_proto_np_list[label], value, axis=0)
    return global_proto_np_list


def get_global_proto_kde(global_proto_np_list, global_proto_kde):
    for label, value in global_proto_np_list.items():
        x = np.vstack(value)
        global_proto_kde[label] = kde(x)
    return global_proto_kde


def get_malicious(client_Proto, global_proto_kde):
    evaluate_value_list = []
    for label, value in client_Proto.items():
        for i in range(len(global_proto_kde[label])):
            evaluate_value_list.append(global_proto_kde[label].evaluate(value[i]))
    evaluate_value = np.mean(evaluate_value_list)
    print(evaluate_value)
    if evaluate_value > 0.5:
        return 1
    else:
        return 0
