# This shows how k-fold can be applied to a dataset. The 'winequality-red.csv'
# dataset will be used again here to demonstrate this.

# k-fold is a method of cross-validation that has a single parameter, 'k',
# that refers to the number of groups that a given data sample is split into.
# This method can be used with almost any kind of data. For example, if a
# dataset is made up of images, a csv file can be created with 'image id', 
# 'image location', and 'image label', and k-fold can be used to split the data.

# Import necessary packages
import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":

    # Read the dataset stored in 'train.csv'
    df = pd.read_csv("winequality-red.csv")

    # Create a new column called 'kfold' and fill it with -1
    df["kfold"] = -1

    # Shuffle the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)

    # Initiate the kfold class from the 'model_selection' module
    kf = model_selection.KFold(n_splits=5)

    # Fill the new kfold column and split the data into 5 sections
    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, 'kfold'] = fold

    # Save the data into a new file 'train_folds.csv'
    df.to_csv("train_folds.csv", index=False)
