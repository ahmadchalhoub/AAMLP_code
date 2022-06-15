# This script shows the implementation of most of the 'error'
# evaluation metrics used in evaluating regression problems.

# Keep in mind that the implementations of the metrics here are not done in
# the most efficient way possible. They can be calculated much more efficiently
# using numpy more. However, looking at the most basic and clear implementation
# of these metrics makes it much easier to study and understand them.

import numpy as np

def mean_absolute_error(y_true, y_pred):
    """
    This function calculates the mean absolute error metric
    :param y_true: list of values, actual classes
    :param y_pred: list of values, predicted classes
    :return: mean absolute error value
    """

    # initialize an error value to 0
    error = 0

    # iterate through true and predicted values 
    # and add absolute error to 'error' variable
    for yt, yp in zip(y_true, y_pred):
        error += np.abs(yt - yp)

    # return ME
    return error / len(y_true)    
        
def mean_squared_error(y_true, y_pred):
    """
    This function calculates the mean squared error metric
    :param y_true: list of values, actual classes
    :param y_pred: list of values, predicted classes
    :return: mean squared error value
    """

    # intialize an error value to 0
    error = 0

    # iterate through true and predicted values 
    # and add squared error to 'error' variable
    for yt, yp in zip(y_true, y_pred):
        error += np.square(yt - yp)

    # return MSE
    return error / len(y_true)

def mean_squared_log_error(y_true, y_pred):
    """
    This function calculates the mean squared log error metric
    :param y_true: list of values, actual classes
    :param y_pred: list of values, predicted classes
    :return: mean squared log error value
    """

    # intialize an error value to 0
    error = 0

    # iterate through true and predicted values 
    # and add squared log error to 'error' variable
    for yt, yp in zip(y_true, y_pred):
        error += np.square((np.log(1 + yt) - np.log(1 + yp)))

    # return MSLE
    return error / len(y_true)

def root_mean_squared_log_error(y_true, y_pred):
    """
    This function calculates the root mean squared log error metric
    :param y_true: list of values, actual classes
    :param y_pred: list of values, predicted classes
    :return: root mean squared log error value
    """

    # intialize an error value to 0
    error = 0

    # iterate through true and predicted values 
    # and add root squared log error to 'error' variable
    for yt, yp in zip(y_true, y_pred):
        error += (np.log(1 + yt) - np.log(1 + yp))

    # return RMSLE
    return error / len(y_true)

def mean_percentage_error(y_true, y_pred):
    """
    This function calculates the mean percentage error metric
    :param y_true: list of values, actual classes
    :param y_pred: list of values, predicted classes
    :return: mean percentage error value
    """

    # intialize an error value to 0
    error = 0

    # iterate through true and predicted values 
    # and add mean percentage error error to 'error' variable
    for yt, yp in zip(y_true, y_pred):
        error += (yt - yp) / yt

    # return Percentage Error
    return error / len(y_true)

def mean_absolute_percentage_error(y_true, y_pred):
    """
    This function calculates the mean absolute percentage error metric
    :param y_true: list of values, actual classes
    :param y_pred: list of values, predicted classes
    :return: mean absolute percentage error value
    """

    # intialize an error value to 0
    error = 0

    # iterate through true and predicted values 
    # and add mean absolute percentage error error to 'error' variable
    for yt, yp in zip(y_true, y_pred):
        error += np.abs((yt - yp)) / yt

    # return MAPE
    return error / len(y_true)

def r_squared(y_true, y_pred):
    """
    This function calculates the r-squared metric
    :param y_true: list of values, actual classes
    :param y_pred: list of values, predicted classes
    :return: r-squared value
    """

    # initialize all parameters in r-squared formula
    top = 0
    bottom = 0
    y_mean = np.mean(y_true)

    # iterate through true and predicted values 
    # and update all variables
    for yt, yp in zip(y_true, y_pred):
        top += np.square(yt - yp)
        bottom += np.square((yt - y_mean))

    # return r-squared value
    return 1 - (top / bottom)

if __name__ == "__main__":

    # lists true and predicted values
    y_true = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    y_pred = [2, 1, 3, 1, 2, 3, 3, 1, 2]

    # calculate and print all error values
    print('Calculated Mean Absolute Error = ', mean_absolute_error(y_true, y_pred))
    print('Calculated Mean Squared Error = ', mean_squared_error(y_true, y_pred))
    print('Calculated Mean Squared Log Error = ', mean_squared_log_error(y_true, y_pred))
    print('Calculated Root Mean Squared Log Error = ', root_mean_squared_log_error(y_true, y_pred))
    print('Calculated Mean Percentage Error = ', mean_percentage_error(y_true, y_pred))
    print('Calculated Mean Absolute Percentage Error = ', mean_absolute_percentage_error(y_true, y_pred))
    print('Calculated R-Squared value = ', r_squared(y_true, y_pred))