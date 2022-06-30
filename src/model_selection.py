from typing import List
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import matthews_corrcoef
from statistics import mean, stdev
import tqdm

import data_analysis
import data_processing

import matplotlib.pyplot as plt
from sklearn import tree

def create_model3(optimizer='rmsprop', init='glorot_uniform'):
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, kernel_initializer=init, activation='relu'))
    model.add(Dense(8, kernel_initializer=init, activation='relu'))
    model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def create_model5(optimizer='rmsprop', init='glorot_uniform'):
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, kernel_initializer=init, activation='relu'))
    model.add(Dense(12, kernel_initializer=init, activation='tanh'))
    model.add(Dense(8, kernel_initializer=init, activation='relu'))
    model.add(Dense(8, kernel_initializer=init, activation='tanh'))
    model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def compare_model_types(X_train, y_train):
    norm_X_train = data_processing.normalize_for_nets(X_train)

    print_cross_val_score(KerasClassifier(create_model3, epochs=1, verbose=0), X_train, y_train, repeat=2, name="create_model3, epochs=1, verbose=0")
    print_cross_val_score(KerasClassifier(create_model3, epochs=2, verbose=0), X_train, y_train, repeat=2, name="create_model3, epochs=2, verbose=0")

    print_cross_val_score(KerasClassifier(create_model5, epochs=1, verbose=0), X_train, y_train, repeat=2, name="create_model5, epochs=1, verbose=0")
    print_cross_val_score(KerasClassifier(create_model5, epochs=2, verbose=0), X_train, y_train, repeat=2, name="create_model5, epochs=2, verbose=0")

    # KerasClassifier(mean: 0.676 - std: 0.061 )
    # KerasClassifier(mean: 0.683 - std: 0.071 )
    # KerasClassifier(mean: 0.683 - std: 0.057 )
    # KerasClassifier(mean: 0.685 - std: 0.062 )

    # KerasClassifier ( mean: 0.669 - std: 0.064 )( mean: 0.665 - std: 0.077
    # KerasClassifier ( mean: 0.636 - std: 0.077 )( mean: 0.633 - std: 0.079
    # KerasClassifier ( mean: 0.674 - std: 0.074 )( mean: 0.676 - std: 0.052
    # KerasClassifier ( mean: 0.674 - std: 0.057 )( mean: 0.671 - std: 0.059
    # KerasClassifier ( mean: 0.673 - std: 0.059 )( mean: 0.682 - std: 0.075


    # print_cross_val_score(LogisticRegression(max_iter=300), X_train, y_train)
    #
    # print_cross_val_score(LogisticRegression(max_iter=2000, C=0.1, penalty="l1", solver="liblinear"), X_train, y_train)
    #
    # print_cross_val_score(DecisionTreeClassifier(
    #         max_depth=4,
    #         min_samples_leaf=6,
    #         min_samples_split=4),
    #     X_train, y_train)
    #
    # print_cross_val_score(RandomForestClassifier(
    #         n_estimators=140,
    #         class_weight="balanced",
    #         max_depth=10,
    #         min_samples_leaf=3,
    #         min_samples_split=5),
    #     X_train, y_train, repeat=30)

def print_cross_val_score(model, data, target, repeat = 1, name = None):
    scores = []
    for _ in tqdm.tqdm(range(repeat), leave=False):
        scores.extend(cross_val_score(model, data, target, scoring="f1", n_jobs=-1))
    print('\r    \r', end='', flush=True)

    scores = list(map(lambda x: round(x, 3), scores))

    if name is None: name = type(model).__name__
    print("{0} ( mean: {1}, std: {2} )".format(name.rjust(28), round(mean(scores), 3), round(stdev(scores), 3))) # , scores

def compare_model_params(model, param_grid, X_train, y_train):
    gs_results = GridSearchCV(model, param_grid=param_grid, cv=4, scoring="f1", verbose=2, n_jobs=-1).fit(X_train, y_train)

    results = pd.DataFrame(gs_results.cv_results_)
    results.loc[:, 'mean_test_score'] *= 100
    # results.to_csv(r"../parameter-analysis/{0}.csv".format(file_name))

    # take the most relevant columns and sort (for readability)
    results = results.loc[:, ('rank_test_score', 'mean_test_score', 'params')]
    results.sort_values(by='rank_test_score', ascending=True, inplace=True)

    return results