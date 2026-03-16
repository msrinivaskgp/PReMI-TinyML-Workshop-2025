from basic_functions import *
import timeit
import numpy as np
from config import Terms, trees, n_class

class RF_Func:
    def dt_predict(dt, winetest):
        arg_dt = []
        arg_dtp = []
        correct = 0

        labels = sorted(list(basic_functions.class_counts(winetest).keys()))
        label_indices = {label: idx for idx, label in enumerate(labels)}
        sum_p = np.zeros(len(labels), dtype=float)

        for t in range(len(winetest)):
            post = []
            for d in range(trees):
                for n in range(len(dt[d][0])):
                    if dt[d][5][n]:
                        feature_idx = dt[d][4][n]
                        threshold = dt[d][5][n]
                        go_left = winetest[t][feature_idx] > threshold

                        child_idx = dt[d][2][n] if go_left else dt[d][3][n]
                        if child_idx and not dt[d][5][child_idx]:
                            post.append(dt[d][8][child_idx])
                            break

            sum_p[:] = 0.0  # reset for each test sample
            for p in post:
                probs, classes = p
                for prob, cls in zip(probs, classes):
                    idx = label_indices[cls]
                    sum_p[idx] += prob

            max_class = sum_p / trees
            arg_max = np.argmax(max_class)

            arg_dt.append(arg_max)
            arg_dtp.append(max_class[arg_max])

            if int(winetest[t][-1]) == arg_max:
                correct += 1

        acc1 = correct / len(winetest)
        print('-------------------------------------------')
        print('RF:', acc1)
        print('-------------------------------------------')
        return acc1, arg_dt

    def dtv_predict(dt, winetest):
        arg_dt1 = []
        correct = 0

        labels = sorted(list(basic_functions.class_counts(winetest).keys()))
        label_indices = {label: idx for idx, label in enumerate(labels)}
        sum_p = np.zeros(len(labels), dtype=float)

        for t in range(len(winetest)):
            post = []
            for d in range(trees):
                for n in range(len(dt[d][0])):
                    if dt[d][5][n]:
                        feature_idx = dt[d][4][n]
                        threshold = dt[d][5][n]
                        go_left = winetest[t][feature_idx] > threshold

                        child_idx = dt[d][2][n] if go_left else dt[d][3][n]
                        if child_idx and not dt[d][5][child_idx]:
                            post.append(dt[d][8][child_idx])
                            break

            votes = []
            for p in post:
                probs, classes = p
                class_max_index = np.argmax(probs)
                votes.append(classes[class_max_index])

            predicted_class = basic_functions.most_frequent(votes)
            arg_dt1.append(predicted_class)

            if int(winetest[t][-1]) == predicted_class:
                correct += 1

        acc2 = correct / len(winetest)
        print('-------------------------------------------')
        print('RF-V:', acc2)
        print('-------------------------------------------')
        return acc2, arg_dt1
