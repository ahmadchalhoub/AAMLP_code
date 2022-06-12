# This script shows how the F1 score can be calculated manually
# from recall and precision, as well as from the Sklearn library.

# F1 score takes both recall and precision into account, and should
# be used when working with datasets that have skewed targets.

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

def true_negative(y_true, y_pred):
    """
    Function to calculate True Negatives
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: number of True Negatives
    """

    # Initialize the number of True Neagtives to 0
    tn = 0

    # Loop through all list entries and increment tn counter
    # for every true negative entry
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
            tn += 1

    # Return the number of true negatives
    return tn

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

def calculate_f1(y_true, y_pred):
    """
    Function to calculate f1 score from targets and predictions
    :param y_true: target values
    :param y_pred: predicted values
    :return: calculated f1 score
    """
    p = calculate_precision(y_true, y_pred)
    r = calculate_recall(y_true, y_pred)
    f1_score = 2*p*r / (p + r)
    return f1_score

if __name__ == "__main__":
    
    # Initialize two arrays, one that has true values
    # and one that has predicted values
    y_true = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
              1, 0, 0, 0, 0, 0, 0, 0, 1, 0]

    y_pred = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
              1, 0, 0, 0, 0, 0, 0, 0, 1, 0]

    # Calculate f1 score using custom functions and calculations
    manual_f1 = calculate_f1(y_true, y_pred)
    print('Manual f1 score = ', manual_f1)

    # Calculate f1 score using sklearn library 
    sklearn_f1 = metrics.f1_score(y_true, y_pred)
    print('Sklearn f1 score = ', sklearn_f1)
