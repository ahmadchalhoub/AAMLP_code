# This script shows how a confusion matrix can be created 
# and graphed using scikit-learn (for creation and calculation)
# and seaborn and matplotlib (for graphing)

# import required libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

# define targets & predictions
y_true = [0, 1, 2, 0, 1, 2, 0, 2, 2]
y_pred = [0, 2, 1, 0, 2, 1, 0, 0, 2]

# get confusion matrix from sklearn
cm = metrics.confusion_matrix(y_true, y_pred)

# graph the confusion matrix
plt.figure(figsize=(10, 10))

# 'cubehelix_palette' produces a colormap with 
# linearly-decreasing, or increasing, brightness.
# More details: https://seaborn.pydata.org/generated/seaborn.cubehelix_palette.html 
cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
sns.set(font_scale=2.5)

# 'heatmap' plots rectangular data as a color-encoded matrix
# More details: https://seaborn.pydata.org/generated/seaborn.heatmap.html
sns.heatmap(cm, annot=True, cmap=cmap, cbar=False)
plt.ylabel('Actual Labels', fontsize=20)
plt.xlabel('Predicted Labels', fontsize=20)
plt.show()