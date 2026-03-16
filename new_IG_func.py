import numpy as np
import random
from basic_functions import *
from typing import List, Any

def tree_table(wine_list1, sublist_size, data, max_feature, max_value, node_classes, parent, left_child, right_child, buff2):

    #global data, max_feature, max_value, node_classes, parent, left_child, right_child, buff2

    data[0], data[1], max_feature[0], max_value[0] = node(wine_list1)
    Minleaf = sublist_size * 0.1

    for i in range(buff2):
        if len(data[i]) > Minleaf:
            data[2 * i + 2], data[2 * i + 3], max_feature[i + 1], max_value[i + 1] = node(data[i])

    tree_node = [wine_list1] + [data[i] for i in range(buff2)]

    for i in range(buff2):
        class_counts = basic_functions.class_counts(tree_node[i])
        node_classes[i] = list(class_counts.items())
        parent[i] = (i - 1) // 2 if i % 2 == 1 else (i - 2) // 2
        if max_value[i]:
            left_child[i] = 2 * i + 1
            right_child[i] = 2 * i + 2

    tree1 = [i for i in range(buff2) if node_classes[i]]
    parent1 = np.where(np.array(tree1) % 2 == 1, (np.array(tree1) - 1) // 2, (np.array(tree1) - 2) // 2)
    parent2 = [tree1.index(p) if p in tree1 else -1 for p in parent1]

    left_child1 = 2 * np.array(tree1) + 1
    right_child1 = 2 * np.array(tree1) + 2
    left_child2 = [tree1.index(lc) if lc in tree1 else -1 for lc in left_child1]
    right_child2 = [tree1.index(rc) if rc in tree1 else -1 for rc in right_child1]

    max_feature2 = [max_feature[i] for i in tree1]
    max_value2 = [max_value[i] for i in tree1]
    node_classes2 = [node_classes[i] for i in tree1]

    class_probs = []
    node_max_class = []
    for node_class in node_classes2:
        counts = np.array([count for _, count in node_class])
        classes = [cls for cls, _ in node_class]
        probs = counts / counts.sum() if counts.sum() != 0 else np.zeros_like(counts)
        class_probs.append([probs.tolist(), classes])
        node_max_class.append(classes[int(np.argmax(probs))])

    return [tree1, parent2, left_child2, right_child2, max_feature2, max_value2, node_classes2, node_max_class, class_probs]


def info_gain(wine_list1):
    wine = np.asarray(wine_list1)
    P, Q = wine.shape
    target = wine[:, -1]
    features = wine[:, :-1]

    nums = list(range(Q - 1))
    random.shuffle(nums)
    num = nums[:Q - 1]

    features_shuffled = features[:, num]
    training = np.column_stack((features_shuffled, target))

    feature_mins = np.min(features, axis=0)
    feature_maxs = np.max(features, axis=0)
    ig = np.random.uniform(
        low=feature_mins[:, np.newaxis],
        high=feature_maxs[:, np.newaxis],
        size=(Q - 1, Q - 1)
    )

    new_size = 2
    N1 = 2
    N = 2
    ig1 = ig[:new_size, :N1]

    classes = list(basic_functions.unique_vals(training, -1))
    count = np.zeros(len(classes))
    p = np.zeros(len(classes))

    for i in range(P):
        if training[i, 0] > 0:
            cls = training[i, -1]
            for j, c in enumerate(classes):
                if cls == c:
                    count[j] += 1

    p = count / len(training)
    Entropy_parent = -np.sum(p * np.log2(p + 1e-12))

    countL = np.zeros((len(classes), N, N1))
    countR = np.zeros((len(classes), N, N1))

    for c_idx, c in enumerate(classes):
        for x in range(P):
            for i in range(N):
                for j in range(N1):
                    if training[x, i] > ig1[i, j]:
                        if training[x, -1] == c:
                            countL[c_idx, i, j] += 1
                    else:
                        if training[x, -1] == c:
                            countR[c_idx, i, j] += 1

    left_count = np.sum(countL, axis=0)
    right_count = np.sum(countR, axis=0)

    eps = 1e-9
    prL = (countL + eps) / (left_count + eps)
    prR = (countR + eps) / (right_count + eps)

    WL = left_count / P
    WR = right_count / P

    Entropy_left = -np.sum(prL * np.log2(prL), axis=0)
    Entropy_right = -np.sum(prR * np.log2(prR), axis=0)

    IG = Entropy_parent - WL * Entropy_left - WR * Entropy_right
    f, g = np.unravel_index(np.argmax(IG), IG.shape)

    feature = num[f]
    max_value = ig1[f, g]
    return [feature, max_value]

def node(wine_list1):
    wine = np.asarray(wine_list1)
    P, Q = wine.shape
    max_feature, max_value = info_gain(wine_list1)

    left = []
    right = []

    for i in range(P):
        if wine[i, max_feature] > max_value:
            left.append(wine[i])
        else:
            right.append(wine[i])

    return left, right, max_feature, max_value
