# This shows how stratified k-fold can be applied to a dataset. 
# I made up a very small sample dataset of 20 binary classification
# samples with a very big skew to show how stratified k-fold deals with skew.

# Stratified k-fold ensures that every fold has the same proportion of class labels.
# This means that whatever metric one chooses to evaluate, similar results
# will be produces across all folds. 

# Import necessary packages
import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":

    # Read the dataset stored in 'train.csv'
    df = pd.read_csv("kfold_data_sample.csv")

    # Create a new column called 'kfold' and fill it with -1
    df["kfold"] = -1

    # Shuffle the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)
    print(df['samples'])

    # Store the 'targets', or labels, of the
    # dataset to use for stratified folding
    y = df.target.values
    
    # Initiate the stratified kfold class from the 'model_selection' module
    kf = model_selection.StratifiedKFold(n_splits=2)

    # Fill the new kfold column and split the data into 2 sections
    for fold, (trn_, val_) in enumerate(kf.split(X=df, y=y)):
        print('fold: ', fold)
        print('trn_: ', trn_)
        print('val_: ', val_)
        df.loc[val_, 'kfold'] = fold

    # Save the data into a new file 'stratified_train_folds.csv'
    df.to_csv("stratified_train_folds.csv", index=False)
    