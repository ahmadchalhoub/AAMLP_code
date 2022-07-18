# This script demonstrates how binning can
# be implemented for feature engineering.

import numpy as np
import pandas as pd
from sklearn import preprocessing

# create a random Pandas DataFrame
# with 2 columns and 100 rows
df = pd.DataFrame(
    np.random.rand(100, 2),
    columns=[f"f_{i}" for i in range(1, 3)]
)

# create 10 bins of the numberical columns
df["f_bin_10"] = pd.cut(df["f_1"], bins=10, labels=False)

# create 100 bins of the numberical columns
df["f_bin_100"] = pd.cut(df["f_1"], bins=100, labels=False)
