from gmm_mml import GmmMml
from basic_functions import *
import numpy as np
from Evaluate_boolean import *
import re 
from pyeda.boolalg.expr import exprvar
from pyeda.inter import *
import pickle
import timeit
from config import Terms, trees, n_class

with open('/home/srinivas/Documents/CTRF-main/CTRF-main/CTRF/CTRF/CTRF/Output/test5.pickle', 'rb') as file:
    winetest = pickle.load(file)

pima = np.asarray(winetest)
[P,Q] = pima.shape
target = pima[:,-1]
pfeatures = pima[:,0:Q-1]

def replace_common_with_mean(list1, list2):
    """
    Replaces the common values between list1 and list2 in list1 with the mean value of list2.

    Args:
    - list1: a list of numbers
    - list2: a list of numbers

    Returns:
    - The modified list1 with common values replaced with the mean value of list2.
    """
    common = set(list1).intersection(set(list2))
    mean = sum(list2) / len(list2)
    for i in range(len(list1)):
        if list1[i] in common:
            list1[i] = mean
    return list1

def remove_consecutive_duplicates(lst):
    new_lst = []
    for i, element in enumerate(lst):
        if i == 0 or element != lst[i-1]:
            new_lst.append(element)
    return new_lst

def obdt4(dt,bf,pfeatures):
    unsupervised=GmmMml(plots=True)
    cluster = [[] for i in range(len(pfeatures.T))]
    for d in range(trees):
        for n in range(len(dt[d][0])):
            if(dt[d][5][n]):
                for f in range(len(pfeatures.T)):
                    if(dt[d][4][n]==f):
                        cluster[dt[d][4][n]].append(dt[d][5][n])

    samples = []
    samples_index = []
    samples_cluster = []
    samples_cluster_mean = []
    for f in range(len(pfeatures.T)):
        #Creating histogram
        #fig, ax = plt.subplots(figsize =(10, 7))
        #ax.hist(cluster[f], bins = 10)
        b = np.array(cluster[f])
        arr = b.reshape(len(cluster[f]),1)
        samples.append(arr)
        unsupervised=unsupervised.fit(arr)
        mixture = unsupervised.predict(arr)
        samples_index.append(mixture)
        comp = basic_functions.unique(mixture)
        sum_p = [ [] for i in range(len(comp)) ]
        for d in range(len(arr)):
            for l in range(len(comp)):
                if(comp[l] == mixture[d]):
                    sum_p[l].append(arr[d])
        samples_cluster.append(sum_p)
        mean_p = []
        for d in range(len(comp)):
            mean_p.append(np.average(sum_p[d]))
        samples_cluster_mean.append(mean_p)

    cluster_lists = [ [] for i in range(len(samples_cluster)) ]
    for i in range(len(samples_cluster)):
        for array in samples_cluster[i]:
            new_list = []
            for item in array:
                new_list.append(item[0])
            cluster_lists[i].append(new_list)

    for t in range(trees):
        for cc in range(0,Terms,4):
            #print(bf[t][cc+1])
            #print(bf[t][cc+2])
            f_repeat = []
            for d in range(len(bf[t][cc+1])):
                #print(Repeat(bf[t][cc+1][d]))
                #print(t)
                f_repeat = basic_functions.Repeat(bf[t][cc+1][d])
                if(len(basic_functions.Repeat(bf[t][cc+1][d]))>0):
                    #print(bf[t][cc+1][d])
                    #print(Repeat(bf[t][cc+1][d]))
                    #print(bf[t][cc+2][d])
                    #print(f_repeat[0])
                    #print(cluster_lists[f_repeat[0]])

                    #for i in range(len(cluster_lists)):
                    common_elements = replace_common_with_mean(bf[t][cc+2][d], cluster_lists[f_repeat[0]][0])
                        #new_list = replace_common_with_mean(list1, list2)
                        #if (common_elements == bf[t][cc+2][d]):
                        #  print('No change')
                        #else:
                        #  print(t)
                    #print(common_elements)
                        #print(t)
                #print(len(bf[t][cc+1]))
                #print(cc)
    return bf

def obdt5(bf, var,Terms,remove_f,remove_v):
      cut_tree = []
      for d in range(len(bf)):
        for cc in range(0, Terms, 4):
            if(bf[d][cc]!=[]):
                  for x in range(len(bf[d][cc+2])):
                    duplicates = set()
                    removed_indices = []
                    for i, value in enumerate(bf[d][cc+2][x]):
                        if value in duplicates:
                            removed_indices.append(i)
                        else:
                            duplicates.add(value)
                    unique_list = list(duplicates)
                    if(len(removed_indices)>0):
                      cut_tree.append(d)
                      #print(d)
                      #if (not(d == d+1)):

                    #print(removed_indices)
                    #print(d)
                    #if(len(cut_tree)<var):
                    for i in sorted(removed_indices, reverse=True):
                        remove_f.append(bf[d][cc+1][x][i])
                        remove_v.append(bf[d][cc+2][x][i])
                        del bf[d][cc][x][i+1]
                        del bf[d][cc+1][x][i]
                        del bf[d][cc+2][x][i]
      return bf, cut_tree, remove_f, remove_v


class obds_Func:
    def predict(dt, mt, winetest):
        correct = 0
        arg2 = []
        count99 = 0
        count111 = 0
        avg_test_time = 0
        for v in range(0.01*len(winetest)):
            count_list = []
            for cc in range(0, Terms, 4):
                count = 0
                for d in range(0, trees - 1):
                    my_list = mt[d][cc]
                    my_list1 = dt[d][4]
                    my_list2 = dt[d][5]
                    my_list3 = dt[d][0]
                    list1 = my_list1
                    list2 = winetest[v][0:Q - 1]

                    # Find the indices in list1 that correspond to indices in list2
                    indices1 = [i for i, x in enumerate(list1) if isinstance(x, int) and x < len(list2)]
                    new_list = [list2[list1[i]] if i in indices1 else x for i, x in enumerate(list1)]

                    indices = [i for i in range(len(my_list1)) if not isinstance(my_list1[i], list) or my_list1[i]]
                    list_f = [my_list1[i] for i in indices]
                    list_v = [my_list2[i] for i in indices]
                    list_n = [my_list3[i] for i in indices]
                    list_r = [new_list[i] for i in indices]
                    bool_list = [x > y for x, y in zip(list_v, list_r)]

                    num = len(list_v)
                    variable_names = ['x[{}]'.format(i) for i in range(num)]
                    alphabet_with_commas = map(exprvar, variable_names)

                    reversed_list = [lst[::-1] for lst in my_list if isinstance(lst, list)]

                    keys = list_n
                    values = variable_names
                    my_dict = dict(zip(keys, values))

                    list1 = []
                    list2 = []
                    for my_list in reversed_list:
                        for element in my_list:
                            for i in range(len(my_list) - 1):
                                literal = "~" + my_dict[my_list[i]] if my_list[i + 1] % 2 == 1 else my_dict[my_list[i]]
                                s = literal if i == 0 else s + " & " + literal
                        list1.append(expr(s))
                        list2.append(s)

                    # Create a Boolean expression by taking the OR of all expressions in list1
                    f1 = Or(*list1)

                    expression = repr(f1)
                    and_clauses = re.findall("And\((.*?)\)", expression)
                    and_lists = [clause.split(", ") for clause in and_clauses]

                    lst = list2
                    new_lst1 = [[literal.strip() for literal in s.split('&')] for s in lst]

                    variable_dict = dict(zip(variable_names, bool_list))

                    start_test = timeit.default_timer()
                    result = Evaluate_Boolean .evaluate_boolean_function(and_lists, variable_dict)
                    stop_test = timeit.default_timer()
                    avg_test_time = avg_test_time + (stop_test - start_test)

                    if result:
                        count += 1

                    f1 = f1.to_binary()
                    count99 += len(re.findall(r"\bAnd\b", str(f1)))
                    count111 += len(re.findall(r"\bOr\b", str(f1)))

                count_list.append(count)

            max_index = count_list.index(max(count_list))
            if max_index == int(winetest[v][-1]):
                correct += 1
            arg2.append(max_index)

        acc4 = correct / len(winetest)
        print('-------------------------------------------')
        print('OBDS:', acc4)
        print('-----------------------------')
        print(avg_test_time)
        return acc4, arg2, count99, count111
