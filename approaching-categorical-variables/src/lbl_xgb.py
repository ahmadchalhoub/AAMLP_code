# This script implements an XGBoost model on Label Encoded data

import pandas as pd

import xgboost as xgb
from sklearn import metrics, preprocessing

def run(fold):
    
    # read dataset
    df = pd.read_csv("../input/cat_train_folds.csv")

    # all columns except for 'id', 'target', and 'kfold' are features
    features = [x for x in df.columns if x not in ["id", "target", "kfold"]]

    # fill all NaN values with NONE and convert all columns to type 'str'
    for col in features:
        df[col] = df[col].astype(str).fillna("NONE")

    # label encode the features
    for col in features:

        # initialize LabelEncoder for each feature column
        lbl = preprocessing.LabelEncoder()

        # fit LabelEncoder on all data
        lbl.fit(df[col])

        # transform all the data
        df[col] = lbl.transform(df[col])

    # get training and validation data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # get training and validation data
    x_train = df_train[features].values
    x_valid = df_valid[features].values

    # initialize xgboost model
    model = xgb.XGBClassifier(
        n_jobs=-1,
        max_depth=7,
        n_estimators=200
    )

    # fit model on training data (ohe)
    model.fit(x_train, df_train.target.values)

    # perform predictions on validation data
    valid_preds = model.predict_proba(x_valid)[:, 1]

    # get ROC AUC score
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

    print(f"Fold = {fold}, AUC = {auc}")

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)