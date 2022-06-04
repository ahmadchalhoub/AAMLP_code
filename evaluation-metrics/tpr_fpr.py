# This script implements the calculation of TPR (True Positive Ratio) 
# and FPR (False Positive Ratio) from target and prediction values.
# It also shows how to graph the ROC, or Receiver Operation Charateristic
# curve, and how to calculate the AUC, or Area Under ROC Curve, metric from
# the ROC curve.

import matplotlib.pyplot as plt
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

def calculate_tpr(y_true, y_pred):
    """
    Function to calculate TPR from targets and predictions
    :param y_true: target values
    :param y_pred: predicted values
    :return: calculated TPR
    """
    return calculate_recall(y_true, y_pred)

def calculate_fpr(y_true, y_pred):
    """
    Function to calculate FPR from targets and predictions
    :param y_true: target values
    :param y_pred: predicted values
    :return: calculated FPR
    """
    false_p = false_positive(y_true, y_pred)
    true_n = true_negative(y_true, y_pred)
    return false_p / (false_p + true_n)

if __name__ == "__main__":
    # Empty lists to store TPR and FPR values
    tpr_list = []
    fpr_list = []

    # Target and predicted values
    y_true = [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
    y_pred = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05, 0.9, 0.5,
              0.3, 0.66, 0.3, 0.2, 0.85, 0.15, 0.99]

    # Handmade thresholds
    thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                  0.8, 0.85, 0.9, 0.99, 1.0]

    # Calculate the TPR and FPR values for all threshold
    # values, and store the results in 'tpr_list' and 'fpr_list'
    for thresh in thresholds:
        temp_pred = [1 if x >= thresh else 0 for x in y_pred]
        temp_tpr = calculate_tpr(y_true, temp_pred)
        temp_fpr = calculate_fpr(y_true, temp_pred)
        tpr_list.append(temp_tpr)
        fpr_list.append(temp_fpr)

    # Plot FPR vs TPR. FPR is on the x-axis and TPR is on the y-axis
    # This curve is known as the Receiver Operating Characteristic (ROC).
    # We calculate the area under this curve, which gives us a metric
    # that is usually used when working with a dataset which has 
    # skewed binary targets. This metric is known as the Area Under Curve,
    # or Area Under ROC Curve, or AUC
    plt.figure(figsize=(7, 7))
    plt.fill_between(fpr_list, tpr_list, alpha=0.4)
    plt.plot(fpr_list, tpr_list, lw=3)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.xlabel('FPR', fontsize=15)
    plt.ylabel('TPR', fontsize=15)

    # Calculate the AUC value using Sklearn
    AUC_values = metrics.roc_auc_score(y_true, y_pred)
    print('AUC values = ', AUC_values)

    plt.show()

    
