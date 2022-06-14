import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import data_analysis

import matplotlib.pyplot as plt
from sklearn import tree

def get_trained_model(df: pd.DataFrame) -> DecisionTreeClassifier:
    target_col = "Survived"
    X = df.loc[:, df.columns != target_col]
    y = df.loc[:, target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    decision_tree = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=4,
        min_samples_leaf=5)

    decision_tree.fit(X_train, y_train)

    data_analysis.analize_model(decision_tree, X_test=X_test, y_test=y_test)

    # plt.figure(figsize=(30, 20))    # Размер окна вывода в дюймах
    # tree.plot_tree(decision_tree,  feature_names=list(X.columns), filled=True, fontsize=10)
    # plt.show()

    return decision_tree