import data_analysis
import data_processing

import json

from typing import List
from collections import Counter, defaultdict
from statistics import mean, stdev
import tqdm

import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import matthews_corrcoef
from sklearn import tree

import matplotlib.pyplot as plt

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

def compare_model_types(models, X_train, y_train, repeat=1):
    def print_cross_val_score(model, data, target, repeat=1, name=None):
        scores = []
        for _ in tqdm.tqdm(range(repeat), leave=False):
            scores.extend(cross_val_score(model, data, target, scoring="f1", n_jobs=-1))
        print('\r    \r', end='', flush=True)

        scores = list(map(lambda x: round(x, 3), scores))

        if name is None: name = type(model).__name__
        print("{0} ( mean: {1}, std: {2} )".format(name.rjust(28), round(mean(scores), 3),
                                                   round(stdev(scores), 3)))  # , scores

    norm_X_train = data_processing.normalize_for_nets(X_train)

    for model in models:
        print_cross_val_score(model, X_train, y_train, repeat=repeat)



def compare_model_params(model, param_grid, X_train, y_train, repeat = 1):
    best_mode_results = defaultdict(lambda: defaultdict(int))

    ncols = 0 if repeat == 1 else None
    verbose = 2 if repeat == 1 else 0
    for _ in tqdm.tqdm(range(repeat), ncols=ncols):
        gs_results = GridSearchCV(model, param_grid=param_grid, cv=4, scoring="f1", verbose=verbose, n_jobs=-1).fit(X_train, y_train)

        for param_type in param_grid.keys():
            best_mode_results[param_type][gs_results.best_params_[param_type]] += 1

    print(json.dumps(gs_results, indent=4))