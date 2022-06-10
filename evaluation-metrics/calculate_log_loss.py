# This script shows how the log loss can be calculated manually
# as well as from the Sklearn library.

# Log loss penalizes a lot more than other metrics.

import numpy as np
from sklearn import metrics

# define an (epsilon) value that will be used to clip probabilities.
# using zeros with log can cause errors in calculations, so it is
# important to add or remove a very small epsilon.
epsilon = 1e-15

# empty list to store all loss values
loss = []

def calculate_log_loss(y_tr, y_prob):
    """
    Function to calculate log loss
    :param y_tr: list of true values
    :param y_prob: list of probability values
    :return: average of all log values
    """
    for yt, yp in zip(y_tr, y_prob):

        # clip values in y_proba to [epsilon, 1-epsilon] range
        yp = np.clip(yp, epsilon, 1 - epsilon)

        # calculate log loss for each prediction and append all values to 'loss'
        temp_loss = -1.0 * (yt * np.log(yp) + (1 - yt) * np.log(1 - yp))
        loss.append(temp_loss)
    
    return np.mean(loss)

# lists containing targets and prediction values
y_true = [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
y_proba = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05, 0.9, 0.5, 
           0.3, 0.66, 0.3, 0.2, 0.85, 0.15, 0.99]

# calculate the log loss using the self-created function
calculated_log_loss = calculate_log_loss(y_true, y_proba)

# calculate the log loss using sklearn
sklearn_log_loss = metrics.log_loss(y_true, y_proba)

print(calculated_log_loss)
print(sklearn_log_loss)