
import re
from pyeda.inter import *
from pyeda.boolalg.expr import exprvar
import numpy as np
from Evaluate_boolean import *
import timeit
from config import Terms, trees, n_class
from pyeda.boolalg.bdd import _NODES

#with open('/home/srinivas/Documents/CTRF-main/CTRF-main/CTRF/CTRF/CTRF/Output/test1.pickle', 'rb') as file:
#    winetest = pickle.load(file)

def save_terms_all(filename, v, tree, cc, and_terms):
    """Append all Boolean AND-terms into ONE file"""
    with open(filename, "a") as f:
        for term in and_terms:
            f.write(
                f"v{v} tree{tree} cc{cc} : " +
                " & ".join(term) + "\n"
            )


def save_inputs_all(filename, v, tree, var_dict):
    """Append inputs for EACH (sample, tree) into ONE file"""
    with open(filename, "a") as f:
        f.write(f"v{v} tree{tree} ")
        for k, val in var_dict.items():
            f.write(f"{k}={int(val)} ")
        f.write("\n")

class bds_Func:

    def predict_bds(dt, bf, winetest):
        pima = np.asarray(winetest)
        [P,Q] = pima.shape

        correct = 0
        arg2 = []
        avg_test_time = 0
        unsat_trees = set(range(trees))

        # For OBDD statistics
        obdd_node_counts = []
        obdd_path_counts = []
        obdd_avg_path_lengths = []

        for v in range(int(len(winetest))):
            count_list = []
            count_s1 = 0
            count_d1 = 0
            count_c1 = 0
            count99 = 0
            count111 = 0

            for cc in range(0, Terms, 4):
                count = 0
                for d in range(0, trees, 1):
                    my_list = bf[d][cc]
                    my_list1 = dt[d][4]
                    my_list2 = dt[d][5]
                    my_list3 = dt[d][0]

                    list1 = my_list1
                    list2 = winetest[v][0:Q-1]
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
                    reversed_list = []
                    for lst in my_list:
                        if isinstance(lst, list):
                            reversed_list.append(lst[::-1])

                    keys = list_n
                    values = variable_names
                    my_dict = dict(zip(keys, values))

                    list1 = []
                    for my_list in reversed_list:
                        s = ""
                        for i in range(len(my_list) - 1):
                            lit = my_dict[my_list[i]]
                            if my_list[i+1] % 2 == 1:
                                lit = "~" + lit
                            s = lit if i == 0 else s + " & " + lit
                        list1.append(expr(s))

                    f1 = Or(*list1)

                    # ✅ Convert to OBDD and gather statistics
                    obdd_f1 = expr2bdd(f1)
                    
                    # Count nodes (DFS postorder ensures no duplicates)
                    obdd_node_counts.append(len(list(obdd_f1.dfs_postorder())))
                    # Compute paths & avg path length
                    paths = list(obdd_f1.satisfy_all())
                    obdd_path_counts.append(len(paths))
                    avg_path_len = sum(len(p) for p in paths) / len(paths) if paths else 0
                    obdd_avg_path_lengths.append(avg_path_len)

                    for term in list1:
                        if term.satisfy_one() is not None:
                            unsat_trees.discard(d)
                            break

                    expression = repr(f1)
                    and_clauses = re.findall("And\((.*?)\)", expression)
                    and_lists = [clause.split(", ") for clause in and_clauses]

                    lst = [str(x) for x in list1]
                    new_lst1 = []
                    for s in lst:
                        literals = [literal.strip() for literal in s.split('&')]
                        new_lst1.append(literals)

                    variable_dict = dict(zip(variable_names, bool_list))

                    start_test = timeit.default_timer()
                    result = Evaluate_Boolean.evaluate_boolean_function(and_lists, variable_dict)
                    stop_test = timeit.default_timer()
                    avg_test_time += (stop_test - start_test)

                    prefix = f"v{v}_tree{d}_cc{cc}"

                    save_terms_all(
                        "bds_terms_all.txt",
                        v=v,
                        tree=d,
                        cc=cc,
                        and_terms=and_lists
                    )

                    # -------------------------------------------------
                    # Save inputs ONCE per sample
                    # -------------------------------------------------
                    if cc == 0:
                        save_inputs_all(
                            "bds_inputs_all.txt",
                            v=v,
                            tree=d,
                            var_dict=variable_dict
                        )

                    if result:
                        count += 1

                    f1 = f1.to_binary()
                    
                    count_s1 =count_s1 + f1.size

                    count_d1 =count_d1 + f1.depth

                    count_c1 =count_c1 + f1.cardinality

                    count99 += len(re.findall(r"\bAnd\b", str(f1)))
                    count111 += len(re.findall(r"\bOr\b", str(f1)))

                count_list.append(count)

            max_index = count_list.index(max(count_list))
            if max_index == int(winetest[v][-1]):
                correct += 1
            arg2.append(max_index)

        print("Unsatisfiable BDS trees:", sorted(unsat_trees))
        print("Total unsatisfiable trees:", len(unsat_trees))
        print('-------------------------------------------')
        print('BDS Accuracy:', correct / len(winetest))
        acc3 = correct / len(winetest)
        print(f"Avg Test Time (ms): {avg_test_time*1000:.4f}")
        time1 = avg_test_time * 1000

        # ✅ Report OBDD statistics
        print("\n===== OBDD Statistics =====")
        #print(f"Average Node Count: {sum(obdd_node_counts)/len(obdd_node_counts):.2f}")
        print(f"Average Node Count: {sum(obdd_node_counts)/len(obdd_node_counts):.2f}")
        print(f"Average Path Count: {sum(obdd_path_counts)/len(obdd_path_counts):.2f}")
        print(f"Average Path Length: {sum(obdd_avg_path_lengths)/len(obdd_avg_path_lengths):.2f}")
        print("===========================")

        path_counts =  sum(obdd_path_counts)/len(obdd_path_counts)
        path_lengths = sum(obdd_avg_path_lengths)/len(obdd_avg_path_lengths)
        node_counts = sum(obdd_node_counts) / len(obdd_node_counts)
        print(node_counts)
        return acc3, arg2, count99, count111, count_s1,count_d1,count_c1,time1, path_counts,path_lengths, node_counts

    def predict_eobds(dt, bf, winetest):
        pima = np.asarray(winetest)
        [P,Q] = pima.shape
        correct = 0
        arg2 = []
        avg_test_time = 0
        unsat_trees = set(range(trees))
        obdd_node_counts = []
        obdd_path_counts = []
        obdd_avg_path_lengths = []
        #winetest = wine1
        for v in range(int(len(winetest))):
            count_list = []
            count1 = 0
            count2 = 0
            count3 = 0
            count4 = 0
            count5 = 0
            count6 = 0
            count7 = 0
            count8 = 0
            count99 = 0
            count100 = 0
            count111 = 0
            count122 = 0
            count_s3 = 0
            count_d3 = 0
            count_c3 = 0
            #num_literals1 = 0
            #num_literals2 = 0
            class1_f = []
            class2_f = []
            class1_fm = []
            class2_fm = []
            var1 = []
            var2 = []
            for cc in range(0, Terms, 4):
                count = 0
                for d in range(0,trees,1):
                    #result = has_empty_lists(bf[d])
                    #if(result == False):
                    my_list = bf[d][cc]

                    my_list1 = dt[d][4]
                    #print(my_list1)
                    my_list2 = dt[d][5]
                    my_list3 = dt[d][0]
                    #my_list4 = [-0.00092,  0.01001]

                    list1 = my_list1
                    list2 = winetest[v][0:Q-1]

                    # Find the indices in list1 that correspond to indices in list2
                    indices1 = [i for i, x in enumerate(list1) if isinstance(x, int) and x < len(list2)]

                    # Create a new list with the values from list2 based on the indices in list1
                    new_list = [list2[list1[i]] if i in indices1 else x for i, x in enumerate(list1)]

                    indices = []

                    for i in range(len(my_list1)):
                        if not isinstance(my_list1[i], list) or my_list1[i]:
                            indices.append(i)
                    list_f = [my_list1[i] for i in indices]
                    list_v = [my_list2[i] for i in indices]
                    list_n = [my_list3[i] for i in indices]
                    list_r = [new_list[i] for i in indices]
                    #print(list_f)
                    #print(list_v)
                    #print(list_n)
                    #print(list_r)
                    bool_list = [x > y for x, y in zip(list_v, list_r)]
                    #print(bool_list)

                    # Example usage
                    #num = len(list_v)
                    #alphabets_str = get_alphabets_str(num)
                    #print(alphabets_str) # Output: 'abcde'

                    #alphabet_list = list(alphabets_str)
                    #alphabet_with_commas = ",".join(alphabet_list)
                    #print(alphabet_with_commas)

                    #alphabet_with_commas = map(exprvar, alphabets_str)

                    #if(mt[d][cc] != []):
                    #  my_list = mt[d][cc]

                    num = len(list_v)
                    variable_names = ['x[{}]'.format(i) for i in range(num)]
                    alphabet_with_commas = ",".join(variable_names)

                    alphabet_with_commas = map(exprvar, variable_names)

                    reversed_list = []
                    for lst in my_list:
                        if isinstance(lst, list):
                            reversed_inner_list = lst[::-1]
                            reversed_list.append(reversed_inner_list)

                    #print(reversed_list)

                    for sub_list in reversed_list:
                        for element in sub_list:
                            #print('')
                            pass
                    keys   = list_n
                    values = variable_names
                    my_dict = dict(zip(keys, values))

                    list1 = []
                    list2 = []
                    for my_list in reversed_list:
                        for element in my_list:
                            for i in range(len(my_list)-1):
                                if my_list[i+1] % 2 == 1: # if next element is odd
                                    #print("~" + my_dict[my_list[i]])
                                    literal = "~" + my_dict[my_list[i]]
                                else:
                                    #print(my_dict[my_list[i]])
                                    literal = my_dict[my_list[i]]

                                if i == 0:
                                    s = literal
                                else:
                                    s += " & " + literal
                        list1.append(expr(s)) # convert s to a Boolean expression using expr()
                        list2.append(s)
                    for term in list1:
                        if term.satisfy_one() is not None:
                            unsat_trees.discard(d)  # safely removes 'd' if it exists
                            break  # no need to check other terms
                    # Create a Boolean expression by taking the OR of all expressions in list1
                    f1 = Or(*list1)
                    minimized = espresso_exprs(f1) if len(f1.inputs) > 2 else [f1]

                    for fm in minimized:
                        fm.to_dnf()
                    
                    obdd_f1 = expr2bdd(fm)
                    obdd_node_counts.append(len(list(obdd_f1.dfs_postorder())))
                    paths = list(obdd_f1.satisfy_all())
                    obdd_path_counts.append(len(paths))
                    avg_path_len = sum(len(p) for p in paths) / len(paths) if paths else 0
                    obdd_avg_path_lengths.append(avg_path_len)

                    # define the input expression as a string
                    expression = repr(fm)

                    # use regular expressions to extract the And clauses
                    and_clauses = re.findall("And\((.*?)\)", expression)

                    # split each And clause into a list of its components
                    and_lists = [clause.split(", ") for clause in and_clauses]

                    # print the resulting list of lists
                    #print(and_lists)
                    #print(list2)

                    lst = list2
                    new_lst1 = []
                    for s in lst:
                        literals = [literal.strip() for literal in s.split('&')]
                        new_lst1.append(literals)

                    #print(new_lst1)

                    values = bool_list
                    variable_names = variable_names
                    variable_dict = dict(zip(variable_names, values))

                    #print(variable_dict)

                    expression = new_lst1
                    variable_values = variable_dict
                    result = Evaluate_Boolean.evaluate_boolean_function(expression, variable_values)
                    print(expression)
                    print(variable_values)

                    expression = and_lists
                    variable_values = variable_dict

                    start_test = timeit.default_timer()
                    result = Evaluate_Boolean.evaluate_boolean_function(and_lists, variable_values)
                    stop_test = timeit.default_timer()

                    avg_test_time = avg_test_time + (stop_test - start_test)

                    if(result == True):
                        count = count+1
                    #print(result)
                    #print(and_lists)
                    #print('------')
                    #for clause in f1.cover:
                    #   num_literals1 += len(clause)
                    #for clause in fm.cover:
                    #    num_literals2 += len(clause)
                    fm = fm.to_binary()
                    #fm = fm.to_binary()

                    count_s3 =count_s3 + fm.size

                    count_d3 =count_d3 + fm.depth

                    count_c3 =count_c3 + fm.cardinality

                    count99 =count99 + len(re.findall(r"\bAnd\b", str(fm)))

                    count111 = count111 + len(re.findall(r"\bOr\b", str(fm)))

                count_list.append(count)
                #print(count)

            max_index = count_list.index(max(count_list))
            #print(max_index)

            if(max_index == int(winetest[v][-1])):
                correct = correct+1
            arg2.append(max_index)
        #print(correct)
        print("Unsatisfiable EOBDS trees:", sorted(unsat_trees))
        print("Total unsatisfiable trees:", len(unsat_trees))
        print('-------------------------------------------')
        print('EOBDS:',correct/len(winetest))
        acc3 = correct/len(winetest)
        print(avg_test_time*1000)
        time3 = avg_test_time*1000
        #print(f1)
        #print(len(re.findall(r"\bAnd\b", str(f1))))
        #print(len(re.findall(r"\bOr\b", str(f1))))
        print('-----------------------------')

        print("\n===== OBDD Statistics =====")

        print(f"Average Node Count: {sum(obdd_node_counts)/len(obdd_node_counts):.2f}")
        print(f"Average Path Count: {sum(obdd_path_counts)/len(obdd_path_counts):.2f}")
        print(f"Average Path Length: {sum(obdd_avg_path_lengths)/len(obdd_avg_path_lengths):.2f}")
        print("===========================")

        path_counts =  sum(obdd_path_counts)/len(obdd_path_counts)
        path_lengths = sum(obdd_avg_path_lengths)/len(obdd_avg_path_lengths)
        node_counts = sum(obdd_node_counts) / len(obdd_node_counts)
        return acc3, arg2, count99, count111,count_s3,count_d3,count_c3,time3,path_counts,path_lengths,node_counts
    
