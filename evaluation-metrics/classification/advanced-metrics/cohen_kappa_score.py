# This implementation very briefly shows the implementation of Cohen's Kappa Score
# using scikit-learn.

# We can see that even though accuracy is high, QWK is less.
# A QWK of 0.85 or higher is considered as very good.

from sklearn import metrics

if __name__ == "__main__":

    y_true = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    y_pred = [2, 1, 3, 1, 2, 3, 3, 1, 2]

    print(metrics.cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    print(metrics.accuracy_score(y_true, y_pred))
