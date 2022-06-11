'''
This script shows how the recall can be calculated
for multi-class classification.

There is three different ways to calculate recall
for multi-class classification:

    - Macro averaged recall: calculate recall for all
    classes individually and then average them

    - Micro averaged recall: calculate class wise true positive
    and false negative and then use that to calculate overall recall

    - Weighted recall: same as macro but in this case, it is weighted
    average depending on the number of items in each class
'''

import numpy as np
from collections import Counter
from sklearn import metrics

def true_positive(y_true, y_pred):
    """
    Function to calculate True Positives
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: number of True Positives
    """

    # Initialize the number of True Positives to 0
    tp = 0

    # Loop through all list entries and increment tp counter
    # for every true positive entry
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1

    # Return the number of true positives
    return tp

def false_negative(y_true, y_pred):
    """
    Function to calculate False Negatives
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: number of False Negatives
    """

    # Initialize the number of False Neagtives to 0
    fn = 0

    # Loop through all list entries and increment tn counter
    # for every false negative entry
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 0:
            fn += 1

    # Return the number of false negatives
    return fn

def macro_recall(y_true, y_pred):
    """
    Function to calculate macro averaged recall
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: macro recall score
    """

    # find the number of classes and 
    # initialize recall to 0
    num_classes = len(np.unique(y_true))
    recall = 0
    
    for class_ in range(num_classes):

        # consider all classes except the current class 
        # as negative, so set all other classes to 0
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # calculate true positive and false negative for current class
        tp = true_positive(temp_true, temp_pred)
        fn = false_negative(temp_true, temp_pred)

        # calculate precision for current class
        temp_recall = tp / (tp + fn)

        # keep adding precision for all classes
        recall += temp_recall

    # calculate and return average precision over all classes
    recall /= num_classes
    return recall

def micro_recall(y_true, y_pred):
    """
    Function to calculate micro averaged recall
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: micro recall score
    """

    # find the number of classes and initialize
    # total tp and total fn variables to 0
    num_classes = len(np.unique(y_true))
    tp = 0
    fn = 0

    for class_ in range(num_classes):

        # consider all classes except the current class 
        # as negative, so set all other classes to 0
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]
        
        # calculate tp and fn for current class and 
        # update the over tp and fn variables
        tp += true_positive(temp_true, temp_pred)
        fn += false_negative(temp_true, temp_pred)

    # calculate and return overall precision
    recall = tp / (tp + fn)
    return recall

def weighted_recall(y_true, y_pred):

    """
    Function to calculate weighted averaged recall
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: weighted recall score
    """

    # find the number of classes
    num_classes = len(np.unique(y_true))

    # create class:sample count dictionary, which
    # looks something like this:
    # {0: 20, 1:15, 2:21}
    class_counts = Counter(y_true)

    # initialize precision to 0
    recall = 0

    for class_ in range(num_classes):

        # consider all classes except the current class 
        # as negative, so set all other classes to 0
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # calculate tp and fn for current class
        tp = true_positive(temp_true, temp_pred)
        fn = false_negative(temp_true, temp_pred)

        # calculate precision of class
        temp_recall = tp / (tp + fn)

        # multiply precision with count of samples in class
        weighted_recall = class_counts[class_] * temp_recall

        # add to overall precision
        recall += weighted_recall

    # calculate overall precision by dividing
    # by the total number of samples
    overall_recall = recall / len(y_true)
    return overall_recall

if __name__ == "__main__":
    
    y_true = [0, 1, 2, 0, 1, 2, 0, 2, 2]
    y_pred = [0, 2, 1, 0, 2, 1, 0, 0, 2]

    # calculate and print all three kinds of multi-class precision 
    # from both self-calculation and sklearn 
    print('Calculated macro average recall: ', macro_recall(y_true, y_pred))
    print('Sklearn macro average recall: ', metrics.recall_score(
        y_true, y_pred, average="macro"), '\n')

    print('Micro average recall: ', micro_recall(y_true, y_pred))
    print('Sklearn micro average recall: ', metrics.recall_score(
        y_true, y_pred, average="micro"), '\n')

    print('Weighted recall: ', weighted_recall(y_true, y_pred))
    print('Sklearn micro average recall: ', metrics.recall_score(
        y_true, y_pred, average="weighted"))
        