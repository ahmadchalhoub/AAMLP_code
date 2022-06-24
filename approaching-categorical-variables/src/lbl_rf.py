# This script implements a Random Forest model with 
# Label Encoding on the 'cat-in-the-dat-ii' dataset

import pandas as pd

from sklearn import metrics
from sklearn import ensemble
from sklearn import preprocessing

def run(fold):

    # read dataset
    df = pd.read_csv("../input/cat_train_folds.csv")

    # all columns except for 'id', 'target', and 'kfold' are features
    features = [x for x in df.columns if x not in ["id", "target", "kfold"]]

    # fill all NaN values with NONE and convert all columns to type 'str'
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    for col in features:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(df[col])
        df.loc[:, col] = lbl.transform(df[col])

    # get training and validation data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # get training and validation data
    x_train = df_train[features].values
    x_valid = df_valid[features].values

    # initialize a random forest model
    model = ensemble.RandomForestClassifier(n_jobs=-1)

    # train model (fit model on training data)
    model.fit(x_train, df_train.target.values)

    # perform predictions on the validation data. the predict_proba()
    # method will return a (120000, 2) tensor, with the first column
    # containing the probabilites that the samples are of class 0, and
    # the second column containing the probabilities that they are of class 1.
    # we will use the probability of 1s
    valid_preds = model.predict_proba(x_valid)[:, 1]

    # get ROC AUC score
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

    print(f"Fold = {fold}, AUC = {auc}")

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)
