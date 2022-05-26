# Testing a decision tree algorithm with the task of predicting the quality
# of red wine, using the 'red wine-quality dataset'. This dataset has 1599 entries,
# each with 11 attributes that determine the quality of red wine on a scale from 
# 0 to 10. 

# There is two different ways that this problem can be looked at: classification and
# regression. 

# Import necessary packages
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn import metrics

# Read dataset from csv file
df = pd.read_csv("winequality-red.csv")

# All of the 'quality' values in the dataset are between (including) 3 and 8.
# To make things simpler, those values are mapped to the values 0 to 5.
quality_mapping = {
    3: 0,
    4: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5
}
df.loc[:, "quality"] = df.quality.map(quality_mapping)

# Shuffle the rows in the dataset. This is always an important step to perform
# on data before training a model on that data
df = df.sample(frac=1).reset_index(drop=True)

# Split the data into a training set (1000) and a testing/validation set (599)
df_train = df.head(1000)
df_test = df.tail(599)

# Choose the columns that the tree classifier will be trained on,
# which are basically the features for the model
cols = ['fixed acidity', 
        'volatile acidity', 
        'citric acid', 
        'residual sugar', 
        'chlorides', 
        'free sulfur dioxide', 
        'total sulfur dioxide', 
        'density', 
        'pH', 
        'sulphates', 
        'alcohol']

# Initialize lists to store the different accuracy measurements
# from various training runs with different model depth numbers
train_accuracies = [0.5]
test_accuracies = [0.5]

# For-loop to train the model over a few different depth values, 
# calculating their accuracy measurements, and storing them for 
# plotting and comparison
for depth in range(1, 25):
    # Initialize a decision tree classifier with depth value
    clf = tree.DecisionTreeClassifier(max_depth=depth)
    
    # Train the model on the features selected and their
    # corresponding mapped qualities
    clf.fit(df_train[cols], df_train.quality)

    # Generate predictions on the training and testing data
    # using the trained tree classifier
    train_predictions = clf.predict(df_train[cols])
    test_predictions = clf.predict(df_test[cols])

    # Calculcate accuracy measurements using the training and
    # testing predictions previously generated
    train_accuracy = metrics.accuracy_score(
        df_train.quality, train_predictions)
    test_accuracy = metrics.accuracy_score(
        df_test.quality, test_predictions)

    # Append the accuracy measurements into their lists
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

# Set the global size of the lbel text on the plots
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)

# Create two plots using matplotlib and seaborn
plt.figure(figsize=(10, 5))   # Set the width and height of plot in inches
sns.set_style("whitegrid")
plt.plot(train_accuracies, label="train accuracy")
plt.plot(test_accuracies, label="test accuracy")
plt.legend(loc="lower right", prop={'size' : 10})
plt.xticks(range(0, 26, 5))
plt.xlabel("max depth", size=12)
plt.ylabel("accuracy", size=12)
plt.show()