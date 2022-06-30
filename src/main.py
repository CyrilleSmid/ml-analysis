import pandas as pd
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import matthews_corrcoef
from sklearn.tree import DecisionTreeClassifier
from statistics import mean

import data_analysis
import data_processing
import model_selection

def predict_and_save(model, file_name, is_neural_net=False) -> None:
    test_df = pd.read_csv(r"../datasets/test.csv", sep=",", header="infer", names=None, encoding="utf-8")
    ids, test_df = data_processing.normalize_for_prediction(test_df)

    if is_neural_net:
        test_df = data_processing.normalize_for_nets(test_df)
        print(test_df)

    pred_df = pd.DataFrame(ids)
    y_pred = model.predict(test_df)

    if is_neural_net:
        y_pred = list(map(lambda x: x[0], (y_pred > 0.5).astype("int32")))

    pred_df["Survived"] = pd.Series(y_pred)
    pred_df.to_csv(r"../datasets/{0}.csv".format(file_name), index=False)

def hyper_param_analysis():
    forest_params = {
        "max_depth" : [3,4,5,6],
        "min_samples_split" : [4, 5, 6],
        "min_samples_leaf" : [6,7,8,9],
        "class_weight" : [{0:1,1:1}, "balanced"]
    }
    gs_results = model_selection.compare_model_params(DecisionTreeClassifier(), forest_params, X_train, y_train)
    for _, row in gs_results.head().iterrows():
        print(row.to_dict())

def create_model(optimizer='rmsprop', init='glorot_uniform'):
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, kernel_initializer=init, activation='relu'))
    model.add(Dense(8, kernel_initializer=init, activation='relu'))
    model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

if __name__ == "__main__":
    df = pd.read_csv(r"../datasets/train.csv", sep=",", header="infer", names=None, encoding="utf-8")
    pd.set_option('display.max_rows', 200)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.precision", 2)

    df = data_processing.clean_training_dataset(df)
    data, target = data_processing.separate_data_and_target(df, "Survived")
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=1, shuffle=True)

    norm_data = data_processing.normalize_for_nets(data)
    norm_X_train, norm_X_test, norm_y_train, norm_y_test = train_test_split(norm_data, target, test_size=0.3, random_state=1, shuffle=True)

    model_selection.compare_model_types(data, target)

    # optimizers = ['rmsprop', 'adam']
    # init = ['glorot_uniform', 'normal', 'uniform']
    # epochs = [50, 100, 150]
    # batches = [5, 10, 20]
    # param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init)
    #
    # model = KerasClassifier(build_fn=create_model, verbose=0)
    # gs_results = model_selection.compare_model_params(model, param_grid, norm_X_train, norm_y_train)
    # for _, row in gs_results.head().iterrows():
    #     print(row[["rank_test_score", "mean_test_score", "params"]].to_dict())

    # random_forest = RandomForestClassifier(
    #     n_estimators=140, class_weight="balanced", max_depth=10,
    #     min_samples_leaf=3, min_samples_split=5)
    #
    # random_forest.fit(X_train, y_train)
    #
    # data_analysis.analize_model(random_forest, X_test, y_test)
    #
    # predict_and_save(random_forest, file_name="optimal_forest_submission")


# 	model.add(Dense(12, input_dim=8, kernel_initializer=init, activation='relu'))
# 	model.add(Dense(8, kernel_initializer=init, activation='relu'))
# 	model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))

# 'params': {'batch_size': 10, 'epochs': 100, 'init': 'uniform', 'optimizer': 'rmsprop'
