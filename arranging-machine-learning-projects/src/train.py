# This performs training using the MNIST data on a decision tree classifier

import os
import config
import joblib
import argparse
import pandas as pd
from sklearn import metrics
from sklearn import tree

def run(fold):

    # read the csv of the data with the folds
    df = pd.read_csv(config.TRAINING_FILE)

    # when we use a dataset for training and validation with kfold, we set which folds
    # we want for training and which ones we want for validation. for example, if we 
    # set fold = 0, then we are saying that we want to use the data with kfold value = 0
    # for vlaidation and use the data in all the other folds for training.
    # this is exactly what we do here
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # drop the label column values and store the actual image pixel data
    # and the fold information in 'x_train'. this is stored as a numpy array.
    # then store the label column values in the 'y_train' numpy array
    x_train = df_train.drop("label", axis=1).values
    y_train = df_train.label.values

    # do the same for the validation data
    x_valid = df_valid.drop("label", axis=1).values
    y_valid = df_valid.label.values

    # initialize a simple decision tree classifier from sklearn
    clf = tree.DecisionTreeClassifier()

    # fit the model on training data. the .fit() method takes two parameters,
    # the first one is 'X', which is an array of shape (n_samples, n_features),
    # and the second one is of shape (n_samples,)
    clf.fit(x_train, y_train)

    # create predictions for validation samples
    preds = clf.predict(x_valid)

    # calculate and print the accuracy
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold = {fold}, Accuracy = {accuracy}")

    # save the model, including fold number and its accuracy
    joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, f"dt_{fold}_{accuracy}"))
    
if __name__ == "__main__":

    # initialize an ArgumentParser class of argparse
    parser = argparse.ArgumentParser()
        
    # add the arguments that we want and their type
    parser.add_argument("--fold", type=int)

    # read the arguments from the command line
    args = parser.parse_args()

    # traing the decision tree on the fold that we passed as an argument
    run(fold=args.fold)
