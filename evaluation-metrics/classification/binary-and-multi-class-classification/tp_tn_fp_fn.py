# This script shows how True Positives, True Negatives, False Positives,
# and False Negatives are calculated. And it shows how different evaluation
# metrics, such as accuracy and precision, can be calculated using those variables.

import matplotlib.pyplot as plt

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


def calculate_accuracy_v2(y_true, y_pred):
    """
    Function to calculate accuracy from targets and predictions
    :param y_true: target values
    :param y_pred: predicted values
    :return: calculated accuracy
    """
    true_p = true_positive(y_true, y_pred)
    false_p = false_positive(y_true, y_pred)
    true_n = true_negative(y_true, y_pred)
    false_n = false_negative(y_true, y_pred)

    return (true_p + true_n) / (true_p + true_n + false_p + false_n)


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

if __name__ == "__main__":

    # Initialize two arrays, one that has true values
    # and one that has predicted values
    l1 = [0, 1, 1, 1, 0, 0, 0, 1]
    l2 = [0, 1, 0, 1, 0, 1, 0, 0]

    # Calculate the accuracy from the above variables
    print('Accuracy: ', calculate_accuracy_v2(l1, l2))

    # Calculate the precision from the above variables
    print('Precision: ', calculate_precision(l1, l2))

    # Calculate the recall from the above variables
    print('Recall: ', calculate_recall(l1, l2))


    # Now let's graph a precision-recall curve. We assume 
    # the below target values and the predicted probability values

    # Target values
    y_t = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
              1, 0, 0, 0, 0, 0, 0, 0, 1, 0]

    # Probability values for a sample being assigned
    # a value of 1
    y_p = [0.02638412, 0.11114267, 0.31620708,
            0.0490937, 0.0191491, 0.17554844,
            0.15952202, 0.03819563, 0.11639273,
            0.079377, 0.08584789, 0.39095342,
            0.27259048, 0.03447096, 0.04644807,
            0.03543574, 0.18521942, 0.05934904,
            0.61977213, 0.33056815]

    # Empty lists to store the precision and recall values
    precisions = []
    recalls = []

    # Assumed threshold values. It is not explained in the book
    # how these values are assumed, so this is something that I
    # will try to research and find an answer to
    thresholds = [0.0490937, 0.05934905, 0.079377,
                0.08584789, 0.11114267, 0.11639273,
                0.15952202, 0.17554844, 0.18521942,
                0.27259048, 0.31620708, 0.33056815,
                0.39095342, 0.61977213]

    # For every threshold value in the 'thresholds' list, calculate
    # the precision and recall values of the targets and predictions
    for i in thresholds:
        temp_prediction = [1 if x >= i else 0 for x in y_p]
        p = calculate_precision(y_t, temp_prediction)
        r = calculate_recall(y_t, temp_prediction)
        precisions.append(p)
        recalls.append(r)

    # Plot the precision-recall curve
    plt.figure(figsize=(7, 7))
    plt.plot(recalls, precisions)
    plt.xlabel('Recall', fontsize=15)
    plt.ylabel('Precision', fontsize=15)
    plt.show()
