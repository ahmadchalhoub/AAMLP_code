import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn import model_selection

def create_folds(data):

    # Create a new column called 'kfold' and fill it with -1
    # and shuffle the rows of data
    data["kfold"] = -1
    data = data.sample(frac=1).reset_index(drop=True)

    # Use Sturge's rule to calculate the number of bins
    num_bins = int(np.floor(1 + np.log2(len(data))))

    # Bin the targets using 'pandas.cut', which is used to
    # segment and sort data values into bins. 'cut' is useful
    # for going from a continuous variable to a categorical 
    # variable. It also supports binning into an equal number 
    # of bins, or a pre-specified array of bins
    data.loc[:, "bins"] = pd.cut(
        data["target"], bins=num_bins, labels=False
    )

    # Initiate the stratified kfold class from the 'model_selection' module
    kf = model_selection.StratifiedKFold(n_splits=5)

    # Fill the new kfold column using the data from 'bins'
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'kfold'] = f

    # Drop the 'bins' column
    data = data.drop("bins", axis=1)
    return data

if __name__ == "__main__":

    # Create a sample dataset with 15000 samples, 100 features (for each sample), 
    # and 1 target (for each sample)
    x, y = datasets.make_regression(
        n_samples=15000, n_features=100, n_targets=1
    )

    # Convert the numpy arrays 'x' and 'y' into a pandas DataFrame
    df = pd.DataFrame(
        x,
        columns=[f"f_{i}" for i in range(x.shape[1])]
    )
    df.loc[:, "target"] = y

    # Call the function 'create_folds()'
    df = create_folds(df)
    print(df)