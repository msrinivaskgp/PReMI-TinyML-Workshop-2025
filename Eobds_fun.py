import numpy as np
import pickle
from pyeda.boolalg.expr import exprvar
from pyeda.inter import *
from Evaluate_boolean import *
import re
import timeit
from config import Terms, trees, n_class

# Load the saved result from the file
#with open('C:/Users/DELL/Downloads/CTRF-main/CTRF-main/CTRF/CTRF/CTRF/Output/test5.pickle', 'rb') as file:
#    winetest = pickle.load(file)

#target = pima[:,-1]
#pfeatures = pima[:,0:Q-1]

class eobds_fun:
    def predict(dt, mt, winetest):
        pima = np.asarray(winetest)
        [P,Q] = pima.shape
        correct = 0
        arg2 = []
        avg_test_time = 0
        for v in range(0.01*len(winetest)):
            count_list = []
            total_metrics = {
                'f1_size': 0, 'fm_size': 0, 'f1_depth': 0, 'fm_depth': 0, 
                'f1_cardinality': 0, 'fm_cardinality': 0, 
                'f1_inputs_len': 0, 'fm_inputs_len': 0, 
                'f1_and_count': 0, 'fm_and_count': 0, 
                'f1_or_count': 0, 'fm_or_count': 0
            }
            class_f = [[] for _ in range(n_class)]
            class_fm = [[] for _ in range(n_class)]

            for cc in range(0, Terms, 4):
                count = 0
                for d in range(trees - 1):
                    my_list = mt[d][cc]
                    my_list1 = dt[d][4]
                    my_list2 = dt[d][5]
                    my_list3 = dt[d][0]

                    list1 = my_list1
                    list2 = winetest[v][0:Q-1]

                    # Find indices and create new list
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

                    #Reverse lists for processing
                    reversed_list = [lst[::-1] for lst in my_list if isinstance(lst, list)]

                    keys = list_n
                    values = variable_names
                    my_dict = dict(zip(keys, values))

                    list1, list2 = [], []
                    for sublist in reversed_list:
                        s = ""
                        for i in range(len(sublist)-1):
                            literal = "~" + my_dict[sublist[i]] if sublist[i+1] % 2 == 1 else my_dict[sublist[i]]
                            s = literal if i == 0 else s + " & " + literal
                        list1.append(expr(s))
                        list2.append(s)

                    f1 = Or(*list1)
                    minimized = espresso_exprs(f1) if len(f1.inputs) > 4 else [f1]

                    for fm in minimized:
                        fm.to_dnf()

                    expression = repr(f1)
                    and_clauses = re.findall("And\((.*?)\)", expression)
                    and_lists = [clause.split(", ") for clause in and_clauses]

                    lst = list2
                    new_lst1 = [[literal.strip() for literal in s.split('&')] for s in lst]

                    variable_dict = dict(zip(variable_names, bool_list))
                    start_test = timeit.default_timer()
                    result = Evaluate_Boolean.evaluate_boolean_function(and_lists, variable_dict)
                    stop_test = timeit.default_timer()
                    avg_test_time = avg_test_time + (stop_test - start_test)

                    if result:
                        count += 1

                    f1 = f1.to_binary()
                    fm = fm.to_binary()

                    total_metrics['f1_size'] += f1.size
                    total_metrics['fm_size'] += fm.size
                    total_metrics['f1_depth'] += f1.depth
                    total_metrics['fm_depth'] += fm.depth
                    total_metrics['f1_cardinality'] += f1.cardinality
                    total_metrics['fm_cardinality'] += fm.cardinality
                    total_metrics['f1_inputs_len'] += len(f1.inputs)
                    total_metrics['fm_inputs_len'] += len(fm.inputs)
                    total_metrics['f1_and_count'] += len(re.findall(r"\bAnd\b", str(f1)))
                    total_metrics['fm_and_count'] += len(re.findall(r"\bAnd\b", str(fm)))
                    total_metrics['f1_or_count'] += len(re.findall(r"\bOr\b", str(f1)))
                    total_metrics['fm_or_count'] += len(re.findall(r"\bOr\b", str(fm)))

                    class_f[cc // 4].append(f1)
                    class_fm[cc // 4].append(fm)

                count_list.append(count)

            max_index = count_list.index(max(count_list))
            if max_index == int(winetest[v][-1]):
                correct += 1
            arg2.append(max_index)

        accuracy = correct / len(winetest)
        print('-------------------------------------------')
        print('EOBDS:', accuracy)
        print('-----------------------------')
        print(avg_test_time)
        return accuracy, class_f, class_fm, total_metrics['f1_and_count'], total_metrics['fm_and_count'], total_metrics['f1_or_count'], total_metrics['fm_or_count'], total_metrics['f1_size'], total_metrics['fm_size'], total_metrics['f1_depth'], total_metrics['fm_depth'], total_metrics['f1_cardinality'], total_metrics['fm_cardinality']
