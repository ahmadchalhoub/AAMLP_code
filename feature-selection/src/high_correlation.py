# This script shows how one would use correlation to remove
# some features from a dataset. In this implementation, a new
# feature 'MedInc_Sqrt' is created by applying the square root
# to the existing 'MedInc' feature. After that, when we calculate
# the correlations matrix, we see that the 'MedInc' and 'MedInc_Sqrt'
# features are very highly correlated, and thus one of them could be removed.

import pandas as pd
import numpy as np 

from sklearn.datasets import fetch_california_housing

# fetch a regression dataset - california housing
data = fetch_california_housing()
X = data["data"]
col_names = data["feature_names"]
y = data["target"]

# convert to Pandas DataFrame
df = pd.DataFrame(X, columns=col_names)

# introduce a highly correlated column
df.loc[:, "MedInc_Sqrt"] = df.MedInc.apply(np.sqrt)

# get correlation matrix
df.corr()
