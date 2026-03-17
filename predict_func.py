import numpy as np
import pickle
from pyeda.inter import *
from pyeda.boolalg.expr import exprvar
from pyeda.boolalg.bdd import _NODES

from Evaluate_boolean import *
from new_IG_func import *
from new_MintermCal import *
from new_RF import *
from bds_fun import *
from Eobds_fun import *
from config import Terms, trees, n_class
import cProfile, pstats, io

def profile_function(func, *args, **kwargs):
    """Profile a specific function and print top 20 slowest calls."""
    pr = cProfile.Profile()
    pr.enable()
    result = func(*args, **kwargs)
    pr.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')  # or 'tottime'
    ps.print_stats(30)  # Show top 30 lines
    print(f"\n--- Profiling Report for {func.__name__} ---")
    print(s.getvalue())
    return result

# Load the saved result from the file

with open('/home/srinivas/Documents/PReMI/output/dt1.pickle', 'rb') as file:
    dt = pickle.load(file)
# Load the saved result from the file
with open('/home/srinivas/Documents/PReMI/output/test1.pickle', 'rb') as file:
    winetest = pickle.load(file)
with open('/home/srinivas/Documents/PReMI/output/bf1.pickle', 'rb') as file:
    bf = pickle.load(file)

pima = np.asarray(winetest)
[P,Q] = pima.shape
target = pima[:,-1]
pfeatures = pima[:,0:Q-1]


def has_empty_lists(list_of_lists):
    for sublist in list_of_lists:
        if isinstance(sublist, list) and len(sublist) == 0:
            return True
    return False

acc1,arg_dt = RF_Func.dt_predict(dt, winetest)
acc2,arg_dt1 = RF_Func.dtv_predict(dt, winetest)
#acc_3,arg_1, and_1, or_1,count_s1,count_d1,count_c1,time1,path_counts1, path_lengths1,node_counts1 = bds_Func.predict_bds(dt,bf, winetest)
#acc_4,arg_2, and_2, or_2,count_s2,count_d2,count_c2,time2,path_counts2, path_lengths2, node_counts2 = bds_Func.predict_obds(dt, mt, winetest)
#acc_5,arg_3, and_3, or_3,count_s3,count_d3,count_c3,time3,path_counts3, path_lengths3, node_counts3 = bds_Func.predict_eobds(dt, mt, winetest)
acc_3,arg_1, and_1, or_1,count_s1,count_d1,count_c1,time1,path_counts1, path_lengths1,node_counts1 = \
    profile_function(bds_Func.predict_bds, dt, bf, winetest)

acc_5,arg_3, and_3, or_3,count_s3,count_d3,count_c3,time3,path_counts3, path_lengths3, node_counts3 = \
    profile_function(bds_Func.predict_eobds, dt, bf, winetest)

#acc4,arg1,and2, or2 = obds_Func.predict(dt, mt, winetest)
acc5,class_f, class_fm, and2, and3, or2, or3,size2,size3,depth2,depth3,card2,card3 = eobds_fun.predict(dt, bf, winetest)

#print(and1-and2,and2-and3,or1-or2,or2-or3)
#print(len(class1_fm),len(class2_fm),len(class3_fm))

hist_node = []
hist_depth = []
hist_leaf = []
for d in range(trees):
    count = 0
    for x in range(0,len(dt[d][0])):
        if(not(dt[d][5][x])):
            count = count+1
    hist_leaf.append(count)
    hist_node.append(len(dt[d][0]))
    hist_depth.append(int(np.log2(max(dt[d][0]))))

bf_total = []
for t in range(trees):
    for cc in range(0,Terms,4):
        #print(bf[t][cc+1])
        #print(bf[t][cc+2])
        f_repeat = []
        for d in range(len(bf[t][cc+1])):
            #print(Repeat(bf[t][cc+1][d])) 
            #print(len(bf[t][cc+1][d]))
            bf_total.append(len(bf[t][cc+1][d]))

print(sum(hist_node)-sum(hist_leaf))
print(sum(bf_total))

var1 = []
var2 = []

# Iterate over each class
for c in range(n_class):
    # Initialize input_vars for var1
    input_vars1 = set()
    # Iterate over each instance in the class
    for d in range(len(class_f[c])):
        # Combine inputs from all classes
        input_vars1 |= set(class_f[c][d].inputs)
    # Append the length of unique input variables to var1
    var1.append(len(input_vars1))

# Iterate over each class
for c in range(n_class):
    # Initialize input_vars for var2
    input_vars2 = set()
    # Iterate over each instance in the class
    for d in range(len(class_fm[c])):
        # Combine inputs from all classes
        input_vars2 |= set(class_fm[c][d].inputs)
    # Append the length of unique input variables to var2
    var2.append(len(input_vars2))
print(sum(var1)-sum(var2))

"""
import csv
#new_data = [acc1,acc2,acc3,acc4,acc5,and1, and2, and3,or1,or2, or3,sum(hist_node)-sum(hist_leaf),sum(var1)-sum(var2),sum(bf_total)] 
new_data = [acc1,acc2,acc3,acc5,and2, and3,or2,or3,size2,size3,depth2,depth3,card2,card3,sum(hist_node)-sum(hist_leaf),sum(bf_total),sum(var1)-sum(var2)]
file_path = 'C:/Users/DELL/Downloads/CTRF-main/CTRF-main/CTRF/CTRF/CTRF/Output/file.csv'  # Replace this with the actual path to your CSV file
with open(file_path, 'a', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(new_data)
"""
import csv
file_path = '/home/srinivas/Documents/PReMI/output/file.csv'  # Replace this with the actual path to your CSV file
with open(file_path, 'a', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow([acc1,acc2,acc_3,acc_5,sum(hist_node)-sum(hist_leaf),sum(var1)-sum(var2),and_1,or_1,and_3,or_3,count_s1,count_d1,count_c1,count_s3,count_d3,count_c3,time1,time3,path_counts1, path_lengths1,path_counts3, path_lengths3,node_counts1,node_counts3])
