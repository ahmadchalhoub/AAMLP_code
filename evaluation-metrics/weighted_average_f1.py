# This script shows how the weighted f1 score
# can be calculated for a multi-class classification problem

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

def false_positive(y_true, y_pred):
    """
    Function to calculate False Positives
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: number of False Positives
    """

    # Initialize the number of False Positives to 0
    fp = 0

    # Loop through all list entries and increment fp counter
    # for every false positive entry
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 1:
            fp += 1

    # Return the number of false positives
    return fp

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

def calculate_precision(y_true, y_pred):
    """
    Function to calculate precision from targets and predictions
    :param y_true: target values
    :param y_pred: predicted values
    :return: calculated precision
    """
    true_p = true_positive(y_true, y_pred)
    false_p = false_positive(y_true, y_pred)
    return (true_p) / (true_p + false_p)

def calculate_recall(y_true, y_pred):
    """
    Function to calculate recall from targets and predictions
    :param y_true: target values
    :param y_pred: predicted values
    :return: calculated recall
    """
    true_p = true_positive(y_true, y_pred)
    false_n = false_negative(y_true, y_pred)
    return (true_p) / (true_p + false_n)

def weighted_f1(y_true, y_pred):
    """
    Function to calculate weighted f1 score
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: weighted f1 score
    """

    num_classes = len(np.unique(y_true))
    class_counts = Counter(y_true)

    # initialize total f1 to 0
    f1 = 0

    for class_ in range(num_classes):

        # consider all classes except the current class 
        # as negative, so set all other classes to 0
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        p = calculate_precision(temp_true, temp_pred)
        r = calculate_recall(temp_true, temp_pred)

        if p + r != 0:
            temp_f1 = 2*p*r / (p + r)
        else:
            temp_f1 = 0

        # multiply f1 with the count of samples in class
        weighted_f1 = class_counts[class_] * temp_f1

        # add to f1 total
        f1 += weighted_f1

    # calculate overall f1 by dividing by
    # the total number of samples
    overall_f1 = f1 / len(y_true)
    return overall_f1


if __name__ == "__main__":
    
    y_true = [0, 1, 2, 0, 1, 2, 0, 2, 2]
    y_pred = [0, 2, 1, 0, 2, 1, 0, 0, 2]

    # calculate and print weighted f1-score 
    # from both self-calculation and sklearn 
    print('Calculated weighted f1 score: ', weighted_f1(y_true, y_pred))
    print('Sklearn weighted f1: ', metrics.f1_score(
        y_true, y_pred, average="weighted"))