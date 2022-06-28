# This script implements a Logistic Regression model
# with One Hot Encoding on the 'US Adult Census' dataset

import pandas as pd

from sklearn import metrics
from sklearn import linear_model
from sklearn import preprocessing

def run(fold):

    # read input dataset
    df = pd.read_csv("../input/adult_folds.csv")
    
    # list of numerical columns    
    num_cols = [
        "fnlwgt",
        "age",
        "capital.gain",
        "capital.loss",
        "hours.per.week"
    ]

    # drop numerical columns
    df = df.drop(num_cols, axis=1)

    # map targets (<=50K and >50K) to 0 and 1
    target_mapping = {
        "<=50K": 0,
        ">50K": 1
    }

    #print("type of income column: ", type(df.income))
    df["income"] = df.income.map(target_mapping)

    # all columns are features except income and kfold columns
    features = [x for x in df.columns if x not in ["kfold", "income"]]

    # fill in all NaN values in selected features columns
    # and convert to data type str
    for feature in features:
        df[feature] = df[feature].astype(str).fillna("NONE")

    # separate training and validation data using 'fold'
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # initialize OneHotEncoder instance
    ohe = preprocessing.OneHotEncoder()

    # concatenate training and validation data
    full_data = pd.concat([df_train[features], df_valid[features]], axis=0)

    # fit OHE object to all data
    ohe.fit(full_data)

    # transform training and validation data using OHE object
    x_train = ohe.transform(df_train[features])
    x_valid = ohe.transform(df_valid[features])

    # initialize linear regression model
    model = linear_model.LogisticRegression()

    # train model. pass training data and target labels as params
    model.fit(x_train, df_train.income.values)

    # predict on validation data using trained model.
    # we will use the probability of 1s
    valid_preds = model.predict_proba(x_valid)[:, 1]

    # calculate ROC AUC score
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)

    print(f"Fold = {fold}, AUC = {auc}")


if __name__ == "__main__":
    for i in range(5):
        run(i)