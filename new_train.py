import matplotlib.pyplot as plt
import numpy as np
import random
import math
import pandas
import timeit
import re
import scipy
import itertools
from sklearn.model_selection import StratifiedKFold
from sklearn import datasets
import pickle
import copy
from pyeda.inter import *
from pyeda.boolalg.expr import exprvar
from typing import List, Any

from new_gmm import GmmMml
from new_MintermCal import *
from new_basic_functions import *
from new_IG_func import *
from Evaluate_boolean import *
from obds_func import *
from config import Terms, trees, n_class
#from gmm_mml import GmmMml

k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True)
df = pandas.read_csv('/home/srinivas/Documents/PReMI/data/L10_pn.csv')
list_df = df.values.tolist()
pima = np.asarray(list_df)
[P, Q] = pima.shape
target = pima[:, -1]
pfeatures = pima[:, 0:Q - 1]
times = 1

output_paths = [f'/home/srinivas/Documents/PReMI/data/Output/{{}}{i + 1}.pickle' for i in range(5)]

for f in range(times):
    k_count = 0
    for train_index, test_index in skf.split(pfeatures, target):
        k_count += 1
        X_train, X_test = pfeatures[train_index], pfeatures[test_index]
        y_train, y_test = target[train_index], target[test_index]

        wine = np.column_stack((X_train, y_train))
        wine_list = wine.tolist()
        wine_test = np.column_stack((X_test, y_test))
        wine_test_list = wine_test.tolist()
        winetest = np.asarray(wine_test_list)

        num_sublists = trees
        sublist_size = int(P * 0.9)

        sublists = [random.choices(wine_list, k=sublist_size) for _ in range(num_sublists)]
        
        start = timeit.default_timer()
        dt = []

        for i in range(trees):
            buff, buff2 = 300000, 150000
            data = [[] for _ in range(buff)]
            max_feature = [[] for _ in range(buff2)]
            max_value = [[] for _ in range(buff2)]
            node_classes = [[] for _ in range(buff2)]
            parent = [[] for _ in range(buff2)]
            left_child = [[] for _ in range(buff2)]
            right_child = [[] for _ in range(buff2)]

            #tree_outputs = tree_table(sublists[i], sublist_size)
            tree_outputs = tree_table(sublists[i], sublist_size, data, max_feature, max_value,node_classes, parent, left_child, right_child, buff2)

            dt.append(tree_outputs)

        with open(output_paths[k_count - 1].format('dt'), 'wb') as f_dt, \
             open(output_paths[k_count - 1].format('test'), 'wb') as f_test, \
             open(output_paths[k_count - 1].format('train'), 'wb') as f_train:
            pickle.dump(dt, f_dt)
            pickle.dump(winetest, f_test)
            pickle.dump(wine, f_train)

for fold in range(5):
    with open(output_paths[fold].format('dt'), 'rb') as file:
        dt = pickle.load(file)

    bf5 = []
    for d in range(trees):
        MinT, MinF, MinV, T = Minterm_cal.minterm_now(dt[d][0], dt[d][4], dt[d][5], dt[d][7], n_class)
        sublist = list(itertools.chain.from_iterable(zip(MinT, MinF, MinV, T)))
        bf5.append(sublist)

    bf = copy.deepcopy(bf5)
    with open(output_paths[fold].format('bf'), 'wb') as file:
        pickle.dump(bf, file)

    var_ob = 0
    remove_f, remove_v = [], []

    while True:
        mt = obdt4(dt, bf, pfeatures)
        bf, cut_tree, remove_f, remove_v = obdt5(mt, 1, Terms, remove_f, remove_v)
        unique_list_tree = list(set(remove_consecutive_duplicates(cut_tree)))
        var_ob += len(unique_list_tree)
        if not unique_list_tree:
            break

    with open(output_paths[fold].format('eo'), 'wb') as file:
        pickle.dump(mt, file)

    globals()[f'remove_f{fold + 1}'] = remove_f
    globals()[f'remove_v{fold + 1}'] = remove_v
    globals()[f'var_ob{fold + 1}'] = var_ob

print(*(len(globals()[f'remove_f{i + 1}']) for i in range(5)))
for i in range(5):
    print(basic_functions.count_unique_elements(globals()[f'remove_f{i + 1}']))
