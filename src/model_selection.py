import pandas as pd
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from sklearn.tree import DecisionTreeClassifier
from statistics import mean

import data_analysis

def build_neural_net(optimizer="rmsprop", dense_dims=32):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=8))
    model.add(tf.keras.layers.Dense(units=10, activation="sigmoid"))
    model.add(tf.keras.layers.Dense(units=10, activation="relu"))
    model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
    model.compile(optimizer=optimizer,
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

def compare(df: pd.DataFrame):
    log_reg_scores = []
    svc_scores = []
    random_forest_scores = []

    target_col = "Survived"
    data = df.loc[:, df.columns != target_col]
    norm_data = (data - data.min()) / (data.max() - data.min())
    target = df.loc[:, target_col]

    model = KerasClassifier(build_neural_net, epochs=1000)
    X_train, X_test, y_train, y_test = train_test_split(norm_data, target, test_size=0.3)
    model.fit(X_train, y_train)
    data_analysis.analize_model(model, X_test=X_test, y_test=y_test)

    # print_cross_val_score(LogisticRegression(max_iter=300), data, target)
    #
    # print_cross_val_score(RandomForestClassifier(
    #     n_estimators=100, criterion="entropy",
    #     max_depth=5, random_state=1), data, target)
    #
    # print_cross_val_score(DecisionTreeClassifier(
    #     criterion="entropy", max_depth=5), data, target)

def print_cross_val_score(model, data, target):
    scores = cross_val_score(model, data, target)
    scores = list(map(lambda x: round(x, 3), scores))
    print(type(model).__name__.rjust(28), "(",  round(mean(scores), 3), ")\t", scores)