# This script shows sample code for calculating accuracy.
# Accuracy calculation can be done either manually (by comparing prediction
# results with ground truth), or using scikit-learn

from sklearn import metrics

# Define function to calculate accuracy manually
def calculate_accuracy(y_true, y_pred):

    # Initialize counter to keep track of the number of correct predictions
    correct_counter = 0

    # Loop over the truth and predicted values and update the correct_counter
    # variable accordingly
    for yt, yp in zip(y_true, y_pred):
        if yt==yp:
            correct_counter += 1 
    
    # Calculate and return the accuracy value
    return correct_counter / len(y_true)


if __name__ == "__main__":

    # Initialize two arrays, one that has true values
    # and one that has predicted values
    l1 = [0, 1, 1, 1, 0, 0, 0, 1]
    l2 = [0, 1, 0, 1, 0, 1, 0, 0]

    # Calculate accuracy using sklearn
    sklearn_accuracy = metrics.accuracy_score(l1, l2)
    print('Accuracy using sklearn:', sklearn_accuracy)

    # Call the 'calculate_accuracy' function to get
    # accuracy value by manual calculation
    calculated_accuracy = calculate_accuracy(l1, l2)
    print('Accuracy using calculation: ', calculated_accuracy)
