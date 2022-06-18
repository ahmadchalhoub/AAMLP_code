# This script analyzes the distribution of the MNIST training dataset
# by plotting a bar graph of the data.

# The result showed a very evenly distributed graph, which means that
# there is no skew in the data, so we can use the accuracy or F1 metrics.

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Read csv file containing training data
df = pd.read_csv(
    "/home/ahmad/dev/AAMLP_code/arranging-machine-learning-projects/input/mnist_train.csv")

# Get the number of occurences for every different class and sort indices in ascending order
label_counts = df['label'].value_counts().to_frame().sort_index(ascending=True)

# Plot the data distribution using a bar graph
label_counts.plot.bar()
plt.show()