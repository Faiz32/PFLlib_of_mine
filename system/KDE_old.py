import copy
import numpy as np
from scipy.stats import gaussian_kde
from collections import defaultdict

class proto4client():
    def __init__(self):
        self.protos = defaultdict(list)
        self.protos_var = defaultdict(list)
        self.protos_skewness = defaultdict(list)
        self.malicious = 0


def if_has_nan(data):
    has_nan = np.any(np.isnan(data))
    return has_nan


def if_has_real(data):
    data = data[~np.isnan(data).any(axis=1)]
    if len(data) == 0:
        return False
    else:
        return True


def check_data(data):
    if if_has_nan(data) and if_has_real(data):
        return 1  # 有nan也有实数
    elif if_has_nan(data) and not if_has_real(data):
        return 2  # 有nan无实数
    elif not if_has_nan(data) and if_has_real(data):
        return 3  # 无nan有实数
    else:
        return 4  # 无nan无实数


def extract_protos_data(proto_list_for_label):
    intensities = []
    for j in range(len(proto_list_for_label)):
        proto_list_for_label_new = proto_list_for_label[j].clone().detach().cpu()
        proto_list_for_label_np = np.array(proto_list_for_label_new, dtype=np.float64)
        intensities.append(proto_list_for_label_np)
    intensities_np = np.vstack(intensities)
    return intensities_np


def proto_kde(proto_list):
    # 将原型列表转置并转换为浮点数类型的numpy数组
    proto_list_T = np.array(proto_list.T, dtype=np.float64)
    # 检查数据的合法性
    # check_data(proto_list_T)
    kde_list = []
    for i in range(len(proto_list_T)):
        # 深拷贝第一个元素，用于后续比较和可能的常数值添加
        a = copy.deepcopy(proto_list_T[i][0])
        # 如果列表中的所有元素都相等，则将该常数值添加到kde_list中
        if np.all(proto_list_T[i] == a):
            kde_list.append(a)
        else:
            # 否则，计算高斯核密度估计并添加到kde_list中
            kde = gaussian_kde(proto_list_T[i])
            kde_list.append(kde)
    return kde_list


def get_protos_bool(proto_list, protos_bool):
    kde_list = proto_kde(proto_list)
    proto_list_evaluate = []
    for i in range(len(proto_list)):
        proto_kde_evaluate_value = []
        for j in range(len(kde_list)):
            if isinstance(kde_list[j], np.float64):
                evaluate_value = 0
                if proto_list[i][j] == kde_list[j]:
                    evaluate_value = 100
                else:
                    evaluate_value = 0
                proto_kde_evaluate_value.append(evaluate_value)
            else:
                evaluate_value = kde_list[j].evaluate(proto_list[i][j])
                proto_kde_evaluate_value.append(evaluate_value[0])
        mean = np.mean(proto_kde_evaluate_value)
        if mean > 21:
            protos_bool[i] = True
        else:
            protos_bool[i] = False
    return protos_bool


def proto_kde_evaluate(proto_list_for_label):
    protos_bool = [False] * len(proto_list_for_label)
    proto_list = extract_protos_data(proto_list_for_label)
    if check_data(proto_list) == 1:
        # 有nan也有实数
        proto_list = proto_list[~np.isnan(proto_list).any(axis=1)]
        protos_bool = get_protos_bool(proto_list,protos_bool)
    elif check_data(proto_list) == 2:
        # 有nan无实数
        for i in range(len(proto_list_for_label)):
            if np.any(np.isinf(proto_list[i])):
                protos_bool[i] = True
    else:
        # 无nan有实数
        protos_bool = get_protos_bool(proto_list,protos_bool)
    # if check_data(proto_list) == 4:#无nan无实数

    # kde_list = proto_kde(proto_list)

    return protos_bool


def protos_kde(agg_protos_label):
    agg_protos_label_kde = defaultdict(list)
    for label, proto_list_for_label in agg_protos_label.items():
        protos_bool = proto_kde_evaluate(proto_list_for_label)
        for j in range(len(protos_bool)):
            if protos_bool[j]:
                agg_protos_label_kde[label].append(proto_list_for_label[j])

    return agg_protos_label_kde
