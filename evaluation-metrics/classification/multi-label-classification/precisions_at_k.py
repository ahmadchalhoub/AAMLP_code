# This script shows how 'Precision at k (P@k)', 
# 'Average Precision at k (AP@k)', and 'Mean Average Precision at k (MAP@k)'
# are calculated for a multi-label classification problem, where a single sample
# may contain multiple classes

"""
- Precision at k (P@k): the number of correctly predicted labels
out of the top K labels. For examples, if we have y_true = [1, 2, 3]
and y_pred = [0, 1, 2], we calculate P@3 by looking at the top 3 labels
(in this case all of them). 2 labels were predicted correcly here
(classes 1 and 2), so P@3 would be = 2 / 3 = 0.67.
To calculate P@2, we would look at the top 2 labels, so y_true = [2, 3]
and y_pred = [1, 2], and P@2 = 1/2 = 0.5.
P@1 would be = 0

- Average Precision at k (AP@k): the average of all P@k. For example, to calculate
AP@3, we would calculate P@1, P@2, and P@3, add them all, and divide by 3.

- Mean Average Precision at k (MAP@k): is just an average of AP@k over all samples
"""

def calculate_pk(y_true, y_pred, k):
    """
    This function calculates precision at k for a singe sample
    :param y_true: list of values, actual classes
    :param y_pred: list of values, predicted classes
    :param k: the value for k
    :return: precision at a given value k
    """

    # if k is 0, return 0. this should never happen, as
    # k should always be >= 1
    if k == 0:
        print('In if-statement')
        return 0

    # we are interested only in top-k predictions
    y_pred = y_pred[:k]
    pred_set = set(y_pred)
    true_set = set(y_true)

    # find common values
    common_values = pred_set.intersection(true_set)

    # return length of common values over k
    return len(common_values) / len(y_pred[:k])

def calculate_apk(y_true, y_pred, k):
    """"
    This function calculates average precision at k for a single sample
    :param y_true: list of values, actual classes
    :param y_pred: list of values, predicted classes
    :param k: the value of k
    :return: average precision at a given k
    """

    # initialize empty list to store all pk values
    pk_values = []
    
    # loop over all k. from 1 to k + 1
    for i in range(1, k + 1):
        # calculate p@i and append to list
        pk_values.append(calculate_pk(y_true, y_pred, i))

    if len(pk_values) == 0:
        return 0

    return sum(pk_values) / len(pk_values)

def calculate_mapk(y_true, y_pred, k):
    """"
    This function calculates mean average precision at k for a single sample
    :param y_true: list of values, actual classes
    :param y_pred: list of values, predicted classes
    :param k: the value of k
    :return: mean average precision at a given k
    """

    # initialize empty list to store all apk values
    apk_values = []

    # loop over all samples
    for i in range(len(y_true)):
        # store apk values for every sample
        apk_values.append(calculate_apk(y_true[i], y_pred[i], k=k)
        )

    # return mean of apk values list
    return sum(apk_values) / len(apk_values)

if __name__ == "__main__":

    # define target values
    y_true = [
        [1, 2, 3],
        [0, 2],
        [1],
        [2, 3],
        [1, 0],
        []
    ]

    # define prediction values
    y_pred = [
        [0, 1, 2],
        [1],
        [0, 2, 3],
        [2, 3, 4, 0],
        [0, 1, 2],
        [0]
    ]

    # calculate AP@k for k = 1, 2, & 3
    for i in range(len(y_true)):
        # perform calculations over k = 1, 2, & 3
        for j in range(1, 4):
            print(
                f"""
                y_true={y_true[i]},
                y_pred={y_pred[i]},
                AP@{j}={calculate_apk(y_true[i], y_pred[i], k=j)}
                """
            )

    # calculate MAP@k for k = 1, 2, 3, & 4
    for i in range(1, 5):
        print('MAP@{i}: ', calculate_mapk(y_true, y_pred, k=i))
