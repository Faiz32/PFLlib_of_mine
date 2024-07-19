import copy
import numpy as np
from scipy.stats import gaussian_kde
from collections import defaultdict


def extract_protos_data(proto_list_for_label):
    intensities = []
    for j in range(len(proto_list_for_label)):
        proto_list_for_label_new = proto_list_for_label[j].clone().detach()
        proto_list_for_label_np = proto_list_for_label_new.cpu().detach().numpy()
        intensities.append(proto_list_for_label_np)
    intensities_np = np.vstack(intensities)
    return intensities_np


# 检查数据中是否存在无穷大或NaN值
def check_data(data):
    has_inf = np.any(np.isinf(data))
    has_nan = np.any(np.isnan(data))
    if has_inf or has_nan:
        raise ValueError("Data contains inf or NaN values.")


def proto_kde(proto_list):
    # 将原型列表转置并转换为浮点数类型的numpy数组
    proto_list_T = np.array(proto_list.T, dtype=float)
    # 检查数据的合法性
    check_data(proto_list_T)
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


def proto_kde_evaluate(kde_list, proto_list, protos_bool):
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


def protos_kde(agg_protos_label):
    print(type(agg_protos_label))
    agg_protos_label_kde = defaultdict(list)
    for label, proto_list_for_label in agg_protos_label:
        print(proto_list_for_label)
        # 分别对不同标签下的原型进行聚合，当前选择KDE密度估计
        # 检查特定标签是否存在于聚合原型标签中，如果存在，则对该标签的原型进行处理和评估
        # proto_list_for_label = agg_protos_label[label]
        protos_bool = [True] * len(proto_list_for_label)
        proto_list = extract_protos_data(proto_list_for_label)
        kde_list = proto_kde(proto_list)
        protos_bool = proto_kde_evaluate(kde_list, proto_list, protos_bool)
        for j in range(len(protos_bool)):
            if protos_bool[j]:
                if label in agg_protos_label_kde:
                    agg_protos_label_kde[label].append(proto_list_for_label[j])
                else:
                    agg_protos_label_kde[label] = [proto_list_for_label[j]]
    return agg_protos_label_kde
