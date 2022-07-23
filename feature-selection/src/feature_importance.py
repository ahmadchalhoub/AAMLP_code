# This script shows how features importance values
# obtained using Random Forest model can be plotted.

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

# fetch a regression dataset; in diabetes data,
# we predict diabetes progression after one year
# based on some features
data = load_diabetes()
X = data["data"]
col_names = data["feature_names"]
y = data["target"]

# initialize the model
model = RandomForestRegressor()

# fit the model
model.fit(X, y)

# get the importance of every features
importances = model.feature_importances_

# get indexes that would sort 'importance' in ascending order
idxs = np.argsort(importances)

# plot all the feature importance values
plt.title('Feature Importances')
plt.barh(range(len(idxs)), importances[idxs], align='center')
plt.yticks(range(len(idxs)), [col_names[i] for i in idxs])
plt.xlabel('Random Forest Feature Importance')
plt.show()