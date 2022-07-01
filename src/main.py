import data_analysis
import data_processing
import model_selection

import pprint

import pandas as pd
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Lambda
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import matthews_corrcoef
from sklearn.tree import DecisionTreeClassifier

def predict_and_save(model, file_name, is_neural_net=False) -> None:
    test_df = pd.read_csv(r"../datasets/test.csv", sep=",", header="infer", names=None, encoding="utf-8")
    ids, test_df = data_processing.normalize_for_prediction(test_df)

    pred_df = pd.DataFrame(ids)
    y_pred = model.predict(test_df)

    if is_neural_net:
        y_pred = list(map(lambda x: x[0], (y_pred > 0.5).astype("int32")))

    pred_df["Survived"] = pd.Series(y_pred)
    pred_df.to_csv(r"../datasets/submissions/{0}.csv".format(file_name), index=False)

def hyper_param_analysis():
    # optimizers = ['adam']
    # init = ['glorot_uniform']
    # epochs = [250,300,350]
    # batches = [20]
    # param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init)
    # model_selection.compare_model_params(model, param_grid, norm_X_train, norm_y_train, repeat=10)
    forest_params = {
        "max_depth" : [3,4,5,6],
        "min_samples_split" : [4, 5, 6],
        "min_samples_leaf" : [6,7,8,9],
        "class_weight" : [{0:1,1:1}, "balanced"]
    }
    model_selection.compare_model_params(DecisionTreeClassifier(), forest_params, X_train, y_train)

def neural_network_builder(optimizer='adam', init='glorot_uniform'):
    # create model
    model = Sequential()

    model.add(BatchNormalization())
    model.add(Dense(12, input_dim=8, kernel_initializer=init, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(12, kernel_initializer=init, activation='tanh'))
    model.add(Dense(8, kernel_initializer=init, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(8, kernel_initializer=init, activation='tanh'))
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

    # decision_tree = DecisionTreeClassifier(
    #         max_depth=4,
    #         min_samples_leaf=6,
    #         min_samples_split=4)
    # decision_tree.fit(X_train, y_train)
    # data_analysis.analize_model(decision_tree, X_test, y_test)

    neural_net = KerasClassifier(build_fn=neural_network_builder, verbose=0, epochs=350, batch_size=20)
    neural_net._estimator_type = "classifier"
    neural_net.target_type_ = "binary"

    log_reg = LogisticRegression(max_iter=2000, C=0.1, penalty="l1", solver="liblinear")
    # log_reg.fit(X_train, y_train)

    random_forest = RandomForestClassifier(
            n_estimators=140,
            class_weight="balanced",
            max_depth=10,
            min_samples_leaf=3,
            min_samples_split=5)
    # random_forest.fit(X_train, y_train)

    models = [("neural", neural_net),
              ("forest", random_forest),
              ("logistic", log_reg)]
    voting_classifier = VotingClassifier(models, voting="soft")
    voting_classifier.fit(X_train, y_train)
    data_analysis.analize_model(voting_classifier, X_test, y_test)

    # predict_and_save(voting_classifier, file_name="voting_classifier_submission")
