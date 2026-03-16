class basic_functions:
    def calculate_classification_metrics(prediction, ground_truth):
        if len(prediction) != len(ground_truth):
            raise ValueError("Prediction and ground truth lists must have the same length.")

        tp = 0  # True Positives
        fp = 0  # False Positives
        tn = 0  # True Negatives
        fn = 0  # False Negatives

        for pred, truth in zip(prediction, ground_truth):
            if pred == 1 and truth == 1:
                tp += 1
            elif pred == 1 and truth == 0:
                fp += 1
            elif pred == 0 and truth == 1:
                fn += 1
            else:
                tn += 1

        accuracy = (tp + tn) / (tp + fp + tn + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)
        sensitivity = recall
        specificity = tn / (tn + fp)

        # Calculate kappa score
        total_instances = tp + fp + tn + fn
        p0 = (tp + tn) / total_instances
        pe = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / (total_instances ** 2)
        kappa_score = (p0 - pe) / (1 - pe)

        return accuracy, precision, recall, f1_score, sensitivity, specificity, kappa_score

    def count_unique_elements(list1):
        # Initialize an empty dictionary to store counts
        element_counts = {}
        
        # Iterate through the list
        for x in list1:
            # Check if the element is already in the dictionary
            if x in element_counts:
                # Increment the count if the element exists
                element_counts[x] += 1
            else:
                # Initialize the count to 1 if the element is new
                element_counts[x] = 1
        
        return element_counts

    def most_frequent(List):
        counter = 0
        num = List[0]

        for i in List:
            curr_frequency = List.count(i)
            if(curr_frequency> counter):
                counter = curr_frequency
                num = i
        return num

    def onelistmaker(n):
        listofones = [1] * n
        return listofones

    def class_counts(rows):
        """Counts the number of each type of example in a dataset."""
        counts = {}  # a dictionary of label -> count.
        for row in rows:
            # in our dataset format, the label is always the last column
            label = row[-1]
            label = int(label)
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        return counts

    def unique_vals(rows, col):
        """Find the unique values for a column in a dataset."""
        return set([row[col] for row in rows])

    def unique(list1):
        # intilize a null list
        unique_list = []
        # traverse for all elements
        for x in list1:
            # check if exists in unique_list or not
            if x not in unique_list:
                unique_list.append(x)
        # print list
        return unique_list

    def unique1(list1):
    # intilize a null list
        unique_list = []
        index_list = []
        count = -1
        # traverse for all elements
        for x in list1:
            count = count+1
            # check if exists in unique_list or not
            if x not in unique_list:
                unique_list.append(x)
            else:
                index_list.append(count)
        # print list
        return index_list

    def Repeat(x):
        _size = len(x)
        repeated = []
        for i in range(_size):
            k = i + 1
            for j in range(k, _size):
                if x[i] == x[j] and x[i] not in repeated:
                    repeated.append(x[i])
        return repeated
