# This script demonstrates how polynomial features can
# be created from original features.

from pickle import FALSE
import numpy as np
import pandas as pd
from sklearn import preprocessing

# create a random Pandas DataFrame
# with 2 columns and 100 rows
df = pd.DataFrame(
    np.random.rand(100, 2),
    columns=[f"f_{i}" for i in range(1, 3)]
)

# initialize polynomial features class object
# for two-degree polynomial features
pf = preprocessing.PolynomialFeatures(
    degree=2,
    interaction_only=False,
    include_bias=FALSE
)

# fit to the features and transform
poly_feats = pf.fit_transform(df)

# create a DataFrame with all the features
num_feats = poly_feats.shape[1]
df_transformed = pd.DataFrame(
    poly_feats,
    columns=[f"f_{i}" for i in range(1, num_feats+1)]
)
