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

# Fetches the required dataset from the sklearn datasets. These are
# images of the MNIST digits (a total of 70000), each with dimensions 
# of 28x28, which is flattened into 784 pixels. 'data' is a 2D array
# of dimensions 70000x748.
# These values, along with their labels, are stored in 'pixel_values' and 'targets'.
data = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)

# 'pixel_values' is a pandas DataFrame. 'targets' is a pandas Series.
pixel_values, targets = data

# Initially the values returned and stored in 'targets' are of type string.
# Convert the string values to type int. 'targets' has shape (70000,)
targets = targets.astype(int)

# Creates the t-SNE transformation. 'n_components' sets the dimension
# of the embedded space. 'random_state' determines the random numbe generator.
# This allows for reproducibility (getting same results at different runtimes).
tsne = manifold.TSNE(n_components=2, random_state=42)

# Fits 'pixel_values' into an embedded space. Only the last 3000 numbers are
# used. 'pixel_values.iloc[:3000, :]' results in a (3000, 784) array. 
# After the transformation is done, the output is a (3000, 2) array.
transformed_data = tsne.fit_transform(pixel_values.iloc[:3000, :])

# Stacks the 'transformed_data' array, which has size (3000, 2) with the last 3000
# entries in the 'targets' array, which ahve size (3000, 1), and converting the result
# into a pandas DataFrame, which has size (3000, 3)
tsne_df = pd.DataFrame(np.column_stack((transformed_data, targets.iloc[:3000])),
columns=["x", "y", "targets"])

# Plots the data. The first line creates a multi-plot grid for plotting conditional
# relationships. The second line applies a plotting function to each facet's subset of data.
grid = sns.FacetGrid(tsne_df, hue="targets", height=8)
grid.map(plt.scatter, "x", "y").add_legend()
plt.show()