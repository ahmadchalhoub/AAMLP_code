'''
This script shows how the precision can be calculated
for multi-class classification.

There is three different ways to calculate precision
for multi-class classification:

    - Macro averaged precision: calculate precision for all
    classes individually and then average them

    - Micro averaged precision: calculate class wise true positive
    and false positive and then use that to calculate overall precision

    - Weighted precision: same as macro but in this case, it is weighted
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

def macro_precision(y_true, y_pred):
    """
    Function to calculate macro averaged precision
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: macro precision score
    """

    # find the number of classes and 
    # initialize precision to 0
    num_classes = len(np.unique(y_true))
    precision = 0
    
    for class_ in range(num_classes):

        # consider all classes except the current class 
        # as negative, so set all other classes to 0
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # calculate true positive and true negative for current class
        tp = true_positive(temp_true, temp_pred)
        fp = false_positive(temp_true, temp_pred)

        # calculate precision for current class
        temp_precision = tp / (tp + fp)

        # keep adding precision for all classes
        precision += temp_precision

    # calculate and return average precision over all classes
    precision /= num_classes
    return precision

def micro_precision(y_true, y_pred):
    """
    Function to calculate micro averaged precision
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: micro precision score
    """

    # find the number of classes and initialize
    # total tp and total fp variables to 0
    num_classes = len(np.unique(y_true))
    tp = 0
    fp = 0

    for class_ in range(num_classes):

        # consider all classes except the current class 
        # as negative, so set all other classes to 0
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]
        
        # calculate tp and fp for current class and 
        # update the over tp and fp variables
        tp += true_positive(temp_true, temp_pred)
        fp += false_positive(temp_true, temp_pred)

    # calculate and return overall precision
    precision = tp / (tp + fp)
    return precision

def weighted_precision(y_true, y_pred):

    """
    Function to calculate weighted averaged precision
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: weighted precision score
    """

    # find the number of classes
    num_classes = len(np.unique(y_true))

    # create class:sample count dictionary, which
    # looks something like this:
    # {0: 20, 1:15, 2:21}
    class_counts = Counter(y_true)

    # initialize precision to 0
    precision = 0

    for class_ in range(num_classes):

        # consider all classes except the current class 
        # as negative, so set all other classes to 0
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # calculate tp and fp for current class
        tp = true_positive(temp_true, temp_pred)
        fp = false_positive(temp_true, temp_pred)

        # calculate precision of class
        temp_precision = tp / (tp + fp)

        # multiply precision with count of samples in class
        weighted_precision = class_counts[class_] * temp_precision

        # add to overall precision
        precision += weighted_precision

    # calculate overall precision by dividing
    # by the total number of samples
    overall_precision = precision / len(y_true)
    return overall_precision

if __name__ == "__main__":
    
    y_true = [0, 1, 2, 0, 1, 2, 0, 2, 2]
    y_pred = [0, 2, 1, 0, 2, 1, 0, 0, 2]

    # calculate and print all three kinds of multi-class precision 
    # from both self-calculation and sklearn 
    print('Calculated macro average precision: ', macro_precision(y_true, y_pred))
    print('Sklearn macro average precision: ', metrics.precision_score(
        y_true, y_pred, average="macro"), '\n')

    print('Micro average precision: ', micro_precision(y_true, y_pred))
    print('Sklearn micro average precision: ', metrics.precision_score(
        y_true, y_pred, average="micro"), '\n')

    print('Weighted precision: ', weighted_precision(y_true, y_pred))
    print('Sklearn micro average precision: ', metrics.precision_score(
        y_true, y_pred, average="weighted"))
        