import numpy as np
from collections import Counter

class new_basic_functions:

    @staticmethod
    def calculate_classification_metrics(prediction, ground_truth):
        prediction = np.asarray(prediction)
        ground_truth = np.asarray(ground_truth)

        if prediction.shape[0] != ground_truth.shape[0]:
            raise ValueError("Prediction and ground truth lists must have the same length.")

        tp = np.sum((prediction == 1) & (ground_truth == 1))
        fp = np.sum((prediction == 1) & (ground_truth == 0))
        tn = np.sum((prediction == 0) & (ground_truth == 0))
        fn = np.sum((prediction == 0) & (ground_truth == 1))

        accuracy = (tp + tn) / (tp + fp + tn + fn)
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        sensitivity = recall
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

        total_instances = tp + fp + tn + fn
        p0 = (tp + tn) / total_instances
        pe = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / (total_instances ** 2)
        kappa_score = (p0 - pe) / (1 - pe) if (1 - pe) != 0 else 0

        return accuracy, precision, recall, f1_score, sensitivity, specificity, kappa_score

    @staticmethod
    def count_unique_elements(list1):
        list1 = np.asarray(list1)
        unique_elements, counts = np.unique(list1, return_counts=True)
        return dict(zip(unique_elements.tolist(), counts.tolist()))

    @staticmethod
    def most_frequent(List):
        List = np.asarray(List)
        if np.issubdtype(List.dtype, np.integer):
            counts = np.bincount(List)
            num = np.argmax(counts)
        else:
            counter = Counter(List)
            num = counter.most_common(1)[0][0]
        return num

    @staticmethod
    def onelistmaker(n):
        return [1] * n

    @staticmethod
    def class_counts(rows):
        rows = np.asarray(rows)
        labels = rows[:, -1].astype(int)
        unique_labels, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique_labels.tolist(), counts.tolist()))

    @staticmethod
    def unique_vals(rows, col):
        rows = np.asarray(rows)
        return set(np.unique(rows[:, col]))

    @staticmethod
    def unique(list1):
        list1 = np.asarray(list1)
        unique_list = np.unique(list1).tolist()
        return unique_list

    @staticmethod
    def unique1(list1):
        list1 = np.asarray(list1)
        _, idx_inverse, counts = np.unique(list1, return_inverse=True, return_counts=True)
        index_list = np.where(counts[idx_inverse] > 1)[0].tolist()
        return index_list

    @staticmethod
    def Repeat(x):
        x = np.asarray(x)
        unique_elements, counts = np.unique(x, return_counts=True)
        repeated = unique_elements[counts > 1].tolist()
        return repeated
