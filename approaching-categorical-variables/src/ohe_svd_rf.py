# This script implements a Random Forest classifier on sparse
# One Hot Encoded data using Singular Value Decomposition (SVD)

import pandas as pd

from scipy import sparse
from sklearn import ensemble, metrics, preprocessing, decomposition

def run(fold):
    
    # read dataset
    df = pd.read_csv("../input/cat_train_folds.csv")

    # all columns except for 'id', 'target', and 'kfold' are features
    features = [x for x in df.columns if x not in ["id", "target", "kfold"]]

    # fill all NaN values with NONE and convert all columns to type 'str'
    for col in features:
        df[col] = df[col].astype(str).fillna("NONE")

    # get training and validation data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # initialize OneHotEncoder from scikit-learn.
    # by default, 'sparse' is set to True
    ohe = preprocessing.OneHotEncoder()

    # fit ohe on training + validation features
    x_train = df_train[features]
    x_valid = df_valid[features]
    full_data = pd.concat([x_train, x_valid], axis=0)
    ohe.fit(full_data)

    # transform training and validation data
    x_train = ohe.transform(x_train)
    x_valid = ohe.transform(x_valid)

    # initialize Truncated SVD. here, we
    # choose to reduce the data to 120 components
    svd = decomposition.TruncatedSVD(n_components=120)

    # fit SVD on full sparse training data
    full_sparse = sparse.vstack((x_train, x_valid))
    svd.fit(full_sparse)

    # transform sparse training data
    x_train = svd.transform(x_train)
    x_valid = svd.transform(x_valid)

    # initialize the random forest model. 'n_jobs' sets
    # the number of jobs to run in parallel. 
    # 'n_jobs = -1' means using all processors
    model = ensemble.RandomForestClassifier(n_jobs=-1)

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