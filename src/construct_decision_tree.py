import pandas as pd
from math import log
import operator


def entropy(data_set):
    """计算熵"""
    num_entries = len(data_set)  # 获取数据长度
    label_counts = {}
    for feat_vec in data_set:  # 统计个数
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1

    ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries  # 计算概率值
        ent -= prob * log(prob, 2)  # 更新熵
    return ent


def split_data_set(data_set, axis, value):
    """根据给定的特征划分数据"""
    ret = []
    for feat_vec in data_set:  # 遍历数据集，选取按axis的当前value特征进行划分的数据集
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis+1:])
            ret.append(reduced_feat_vec)

    return ret


def choose_best_feat(data_set):
    """选取最好的特征"""
    num_feat = len(data_set[0]) - 1  # 获取特征个数
    base_ent = entropy(data_set)  # 计算熵
    best_gain = 0.0
    best_feat = -1
    for i in range(num_feat):
        feat_list = [example[i] for example in data_set]
        unique_vals = set(feat_list)
        new_ent = 0.0
        for value in unique_vals:
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set) / float(len(data_set))
            new_ent += prob * entropy(sub_data_set)
        gain = base_ent - new_ent  # 信息增益gain
        if gain > best_gain:  # 选取信息增益最大的特征值
            best_gain = gain
            best_feat = i

    return best_feat


def majority_count(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(data_set, labels):
    class_list = [example[-1] for example in data_set]
    if class_list.count(class_list[0]) == len(class_list):  # 当剩下的数据的类别完全相同时停止划分
        return class_list[0]  # 直接返回标签
    if len(data_set[0]) == 1:  # 遍历完所有特征时
        return majority_count(class_list)  # 返回出现次数最多的类别
    best_feat = choose_best_feat(data_set)
    best_feat_label = labels[best_feat]

    my_tree = {best_feat_label: {}}  # 用字典描述决策树
    del(labels[best_feat])  # 删除已经选过的特征
    feat_vals = [example[best_feat] for example in data_set]
    unique_vals = set(feat_vals)
    for value in unique_vals:
        sub_labels = labels[:]
        my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), sub_labels)
    return my_tree


def classify(input_tree, feat_labels, test_vec):
    """分类"""
    first_str = list(input_tree.keys())[0]
    second_dict = input_tree[first_str]
    feat_index = feat_labels.index(first_str)
    key = test_vec[feat_index]
    value_of_feat = second_dict[key]
    if isinstance(value_of_feat, dict):
        class_label = classify(value_of_feat, feat_labels, test_vec)
    else:
        class_label = value_of_feat
    return class_label


def store_tree(input_tree, file_name):
    import pickle
    fw = open(file_name, 'wb')
    fw.write(pickle.dumps(input_tree))
    fw.close()


def grab_tree(file_name):
    import pickle
    fr = open(file_name, 'rb')
    data = pickle.loads(fr.read())
    fr.close()
    return data


def making_tree():
    csv_path = '../data_set/train.csv'
    img_data = pd.read_csv(csv_path)
    x = []
    for i in range(len(img_data)):
        lst = []
        for j in range(0, 8):
            sub = str(img_data.loc[i, 'Row_'+str(j)])
            lk = str(sub).split(',')
            lst = lst + lk
        lst.append(img_data.loc[i, 'Label'])
        x.append(lst)
    y = []
    for i in range(0, 64):
        y.append('Row_'+str(i))

    lenses_tree = create_tree(x, y)
    store_tree(lenses_tree, '../model/tree.txt')
    print('creating finished!')




