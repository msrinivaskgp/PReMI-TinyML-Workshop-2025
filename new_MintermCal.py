import numpy as np

class Minterm_cal:
    def minterm_now(tree, max_feature, max_value, node_max_class, n_classes):
        tree = np.asarray(tree)

        count = [0] * n_classes
        MinT = [[] for _ in range(n_classes)]
        C = [[] for _ in range(n_classes)]
        T = [[] for _ in range(n_classes)]
        MinF = [[] for _ in range(n_classes)]
        MinV = [[] for _ in range(n_classes)]

        # Determine leaf nodes where max_value is falsy (0 or False)
        leaf_indices = [i for i, val in enumerate(max_value) if not val]

        for idx in leaf_indices:
            class_idx = node_max_class[idx]
            count[class_idx] += 1
            C[class_idx] = []

            node = tree[idx]
            T[class_idx].append(node)

            # Trace parent path for the current node
            W = node
            path = []
            while W != 0:
                parent = (W - 1) // 2 if W % 2 == 1 else (W - 2) // 2
                path.append(parent)
                W = parent

            C[class_idx] = path[::-1]  # Reverse to get path from root to node

            if count[class_idx] > len(MinT[class_idx]):
                MinT[class_idx].append([])

            MinT[class_idx][count[class_idx] - 1] = C[class_idx][:]  # Copy path

            # Extract corresponding features and values for the minterm path
            MinF[class_idx].append([])
            MinV[class_idx].append([])

            for node_in_path in MinT[class_idx][count[class_idx] - 1]:
                for j in range(len(tree)):
                    if node_in_path == tree[j]:
                        MinF[class_idx][count[class_idx] - 1].append(max_feature[j])
                        MinV[class_idx][count[class_idx] - 1].append(max_value[j])

            # Reverse for your pipeline consistency
            MinT[class_idx][count[class_idx] - 1].reverse()
            MinF[class_idx][count[class_idx] - 1].reverse()
            MinV[class_idx][count[class_idx] - 1].reverse()

        # Insert T values into MinT at the start
        for i in range(n_classes):
            for j in range(len(T[i])):
                MinT[i][j].insert(0, T[i][j])

        return MinT, MinF, MinV, T
