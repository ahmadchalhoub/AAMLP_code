import pandas as pd
from sklearn import preprocessing

if __name__ == "__main__":

    # read training and testing data
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')

    # the testing dataset does not have a 'target' 
    # column, so create a fake one
    test.loc[:, "target"] = -1

    # concatenate both training and testing data
    data = pd.concat([train, test]).reset_index(drop=True)

    # make a list of features we are interested in.
    # 'id' and 'target' is something we should not encode
    features = [x for x in train.columns if x not in ["id", "target"]]

    # loop over the features list
    for feat in features:

        # create a new instane of LabelEncoder for each feature
        lbl_enc = preprocessing.LabelEncoder()

        # create a new column that replicates the column of 
        # the feature we are currently on, where all the NaNs 
        # are filled, and all data is converted to a string. 
        # so no matter if it's an int or a float, it is converted
        # to a string and is categorical now. 
        # this is because the encoder requires all elements in a column
        # to be of the same type. if we don't do this, we might have
        # float and NaN in the same column, for example, which wouldn't work
        # PS: the '.values' at the end returns only the values 
        # in the DataFrame; the axes labels get removed
        temp_col = data[feat].fillna("NONE").astype(str).values

        # perform label encoding on the new column using fit_transform
        # and store the results in the original feature column's position
        data.loc[:, feat] = lbl_enc.fit_transform(temp_col)

    # split the training and test data again
    train = data[data.target != -1].reset_index(drop=True)
    test = data[data.target == -1].reset_index(drop=True)
