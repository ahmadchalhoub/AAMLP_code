

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


def calculate_accuracy_v2(tp, fp, tn, fn):
    """
    Function to calculate accuracy from tp, fp, tn, and fn values
    :param tp: true positives
    :param fp: false positives
    :param tn: true negatives
    :param fn: false negatives
    :return: calculated accuracy
    """

    return (true_p + true_n) / (true_p + true_n + false_p + false_n)


def calculate_precision(tp, fp):
    """
    Function to calculate precision from tp and fp values
    :param tp: true positives
    :param fp: false positives
    :return: calculated precision
    """
    return (tp) / (tp + fp)

if __name__ == "__main__":

    # Initialize two arrays, one that has true values
    # and one that has predicted values
    l1 = [0, 1, 1, 1, 0, 0, 0, 1]
    l2 = [0, 1, 0, 1, 0, 1, 0, 0]

    # Call all functions defined above to calculate tp, fp, tn, and fn
    true_p = true_positive(l1, l2)
    print('True positives: ', true_p)
    true_n = true_negative(l1, l2)
    print('True negatives: ', true_n)
    false_p = false_positive(l1, l2)
    print('False positives: ', false_p)
    false_n = false_negative(l1, l2)
    print('False negatives: ', false_n)

    # Calculate the accuracy from the above variables
    print('Accuracy: ', calculate_accuracy_v2(true_p, false_p, true_n, false_n))

    # Calculate the precision from the above variables
    print('Precision: ', calculate_precision(true_p, false_p))
