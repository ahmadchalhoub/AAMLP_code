# This script is a model dispatchet, where we have a dictionary
# that includes keys that are names of models and values that
# are the models themselves. We can then call this dictionary
# from the 'train.py' script the test with different models without 
# having to constantly define and change them in the main script

from sklearn import tree, ensemble

models = {
    "decision_tree_gini" : tree.DecisionTreeClassifier(
        criterion="gini"
    ),
    "decision_tree_entropy" : tree.DecisionTreeClassifier(
        criterion="entropy"
    ),
    "rf" : ensemble.RandomForestClassifier(),
}