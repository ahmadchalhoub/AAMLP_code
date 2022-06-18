# This script performs k-fold on the dataset.
# We are using 5 folds for this dataset 

# Import necessary libraries
import pandas as pd
from sklearn import model_selection

# read csv file containing training data
df = pd.read_csv(
    "/home/ahmad/dev/AAMLP_code/arranging-machine-learning-projects/input/mnist_train.csv")

# add new column 'kfold' and fill it with -1
df['kfold'] = -1

# Shuffle the rows and reset the indices
df = df.sample(frac=1).reset_index(drop=True)

# initialize kfold model with 5 splits
kf = model_selection.KFold(n_splits=5)

# Fill the new kfold column and split the data into 5 sections
for fold, (trn_, val_) in enumerate(kf.split(X=df)):
    df.loc[val_, 'kfold'] = fold

# Write the results into an output file 'mnist_train_folds.csv'
df.to_csv(
    "/home/ahmad/dev/AAMLP_code/arranging-machine-learning-projects/input/mnist_train_folds.csv")
