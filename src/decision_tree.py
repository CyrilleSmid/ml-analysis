import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

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

    y_pred = decision_tree.predict(X_test)

    print("Confusion Matrix:\n",
          confusion_matrix(y_test, y_pred))

    print("Accuracy:\n",
          accuracy_score(y_test, y_pred) * 100)

    print("Report:\n",
          classification_report(y_test, y_pred))

    # plt.figure(figsize=(30, 20))    # Размер окна вывода в дюймах
    # tree.plot_tree(decision_tree,  feature_names=list(X.columns), filled=True, fontsize=10)
    # plt.show()

    return decision_tree