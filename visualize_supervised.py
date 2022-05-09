# Sometimes, when working with data for a supervised ML problem, it is
# beneficial to visualize the data. This can be done by converting the
# data from supervised to unsupervised and performing a clustering technique,
# such as t-distribution Stochastic Neighbor Embedding (t-SNE).

# This script shows how that is applied to the MNIST dataset

import matplotlib.pyplot as plt
import numpy as np                  # used to handle numerical arrays
import pandas as pd                 # used to create dataframes for data
import seaborn as sns               # used for statistical data visualization

from sklearn import datasets
from sklearn import manifold

# Fetches the required dataset from the sklearn datasets. these are
# images of the MNIST digits (a total of 70000), each with dimensions 
# of 28x28, which is flattened into 784 pixels. 'data' is a 2D array
# of dimensions 70000x748.
# These values, along with their labels, are stored in 'pixel_values' and 'targets'
data = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
pixel_values, targets = data

# Initially the values returned and stored in 'targets' are of type string.
# Convert the string values to type int
targets = targets.astype(int)

tsne = manifold.TSNE(n_components=2, random_state=42)
transformed_data = tsne.fit_transform(pixel_values.iloc[:3000, :])

tsne_df = pd.DataFrame(np.column_stack((transformed_data, targets.iloc[:3000])),
columns=["x", "y", "targets"])

tsne_df.loc[:, "targets"] = tsne_df.targets.astype(int)

grid = sns.FacetGrid(tsne_df, hue="targets", size=8)
grid.map(plt.scatter, "x", "y").add_legend()

plt.show()