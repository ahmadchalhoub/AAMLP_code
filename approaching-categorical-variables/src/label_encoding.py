# This script shows how Label Encoding can be done to the training
# data in the 'cat-in-the-dat-ii' dataset. Label Encoding is basically 
# the mapping of text strings into numbers in order to feed that data
# into the models that are later built

import pandas as pd
from sklearn import preprocessing 

if __name__ == "__main__":

    # read training data csv file
    df = pd.read_csv("../input/train.csv")
    df_manual = df.copy()
    df_sklearn = df.copy()

    ############################################################
    # This section shows how to do the Label Encoding manually

    # create dictionary to map text to numbers
    mapping = {
        "Freezing" : 0,
        "Warm" : 1,
        "Cold" : 2, 
        "Boiling Hot" : 3,
        "Hot" : 4,
        "Lava Hot" : 5    
    }

    # perform mapping on the ord_2 text data
    df_manual.loc[:, "ord_2"] = df.ord_2.map(mapping)
    ############################################################

    ############################################################
    # This section shows how to do the Label Encoding using sklearn
    
    # fill NaN values in the ord_2 column
    df_sklearn.loc[:, "ord_2"] = df_sklearn.ord_2.fillna("NONE")
    
    # initialize LabelEncoder
    lbl_enc = preprocessing.LabelEncoder()

    # fit the label encoder and transform values on ord_2 column
    df_sklearn.loc[:, "ord_2"] = lbl_enc.fit_transform(df_sklearn.ord_2.values)
    ############################################################

    # print results to check the outcome
    print(df_manual.loc[:, "ord_2"])
    print(df_manual['ord_2'].unique())
    print(df_sklearn.loc[:, "ord_2"])
    print(df_sklearn['ord_2'].unique())