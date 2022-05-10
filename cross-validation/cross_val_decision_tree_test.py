# Testing a decision tree algorithm with the task of predicting the quality
# of red wine, using the 'red wine-quality dataset'. This dataset has 1599 entries,
# each with 11 attributes that determine the quality of red wine on a scale from 
# 0 to 10. 

# There is two different ways that this problem can be looked at: classification and
# regression. 

# Imports necessary packages and reads dataset from csv file
import pandas as pd
from sklearn import tree
from sklearn import metrics

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

# Shuffles the rows in the dataset. This is always an important step to perform
# on data before training a model on that data
df = df.sample(frac=1).reset_index(drop=True)

# Splits the data into a training set (1000) and a testing/validation set (599)
df_train = df.head(1000)
df_test = df.tail(599)

# Initializing a decision tree classifier with a max depth = 3
clf = tree.DecisionTreeClassifier(max_depth=3)
