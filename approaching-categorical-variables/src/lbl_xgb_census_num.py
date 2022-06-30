# This script implements an XGBoost model with Label Encoded
# data and numerical features on the US Adult Census dataset

import pandas as pd
import xgboost as xgb

from sklearn import preprocessing
from sklearn import metrics

def run(fold):

    # read the input dataset
    df = pd.read_csv("../input/adult_folds.csv")

    # list columns that will be dropped
    num_columns = [
        "fnlwgt",
        "age",
        "capital.gain",
        "capital.loss",
        "hours.per.week"
    ]

    # create dictionary to map binary 
    # 'income' labels to 0 and 1
    income_mapping = {
        "<=50K": 0,
        ">50K": 1
    }
    
    # map labels to 0 and 1
    df["income"] = df.income.map(income_mapping)

    features = [x for x in df.columns if x not in ["kfold", "income"]]

    # convert all data to type 'str'
    for feature in features:
        if feature not in num_columns:
            df[feature] = df[feature].astype(str).fillna("NONE")

    
    # perform Label Encoding on all columns, except 'income' and 'kfold'
    for feature in features:
        if feature not in num_columns:
            lbl = preprocessing.LabelEncoder()
            df[feature] = lbl.fit_transform(df[feature])

    # get training and validation data from folds
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # intialize XGBoost model
    model = xgb.XGBClassifier(
        n_jobs=-1
    )

    # train model
    model.fit(df_train[features].values, df_train.income.values)

    # use trained model to generate predictions on validation data
    predictions = model.predict_proba(df_valid[features].values)[:, 1]

    # calculate ROC AUC metric
    auc = metrics.roc_auc_score(df_valid.income.values, predictions)

    print(f"Fold = {fold}, AUC = {auc}")


if __name__ == "__main__":
    for i in range(5):
        run(i)