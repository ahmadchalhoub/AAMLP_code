import pandas as pd

from sklearn import metrics
from sklearn import preprocessing
from sklearn import linear_model

def run(fold):

    # read dataset
    df = pd.read_csv("../input/cat_train_folds.csv")

    # all columns except for 'id', 'target', and 'kfold' are features
    features = [x for x in df.columns if x not in ["id", "target", "kfold"]]

    # fill all NaN values with NONE and convert all columns to type 'str'
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)
    
    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # initialize OneHotEncoder from scikit-learn
    ohe = preprocessing.OneHotEncoder()

    # fit ohe on training + validation features
    full_data = pd.concat([df_train[features], df_valid[features]], axis = 0)
    ohe.fit(full_data[features])

    # transform training and validation data
    x_train = ohe.transform(df_train[features])
    x_valid = ohe.transform(df_valid[features])

    # initialize Logistic Regression model
    model = linear_model.LogisticRegression()

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