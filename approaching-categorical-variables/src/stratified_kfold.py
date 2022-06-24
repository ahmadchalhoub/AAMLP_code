# To work with the cat-in-the-dat-ii dataset, we
# use Stratified K-Fold to perform cross-validation,
# since it is a binary classification problem with skewed data

import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":

    # read the dataset
    df = pd.read_csv("../input/train.csv")

    # create a 'kfold' column
    df["kfold"] = -1

    # randomize the rows
    df = df.sample(frac=1).reset_index(drop=True)

    # obtain the label values
    y = df.target.values

    # initialize a StratifiedKFold model
    kf = model_selection.StratifiedKFold(n_splits=5)

    # fill the 'kfold' column
    for fold, (trn_, val_) in enumerate(kf.split(X=df, y=y)):
        df.loc[val_, 'kfold'] = fold

    # save the new csv file 
    df.to_csv("../input/cat_train_folds.csv", index=False)
    