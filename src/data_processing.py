import pandas as pd
import numpy as np
import math
from typing import List

import data_analysis

def clean_training_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

    df = clean_missing(df)
    df = clean_outliers(df)
    df = notmalize_categorical(df)

    df_dedupped = df.drop_duplicates()
    return df_dedupped

def normalize_for_prediction(df: pd.DataFrame) -> (List[int], pd.DataFrame):
    ids = df["PassengerId"]
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
    df = clean_missing(df)
    df = notmalize_categorical(df)
    return ids, df

def notmalize_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df["Sex"] = np.where(df["Sex"] == "female", 1,0)

    dummies = pd.get_dummies(df["Embarked"], drop_first=True)
    dummies.rename(columns={"Q": "From_Queenstown", "S": "From_Southampton"}, inplace=True)

    df = pd.concat([df, dummies], axis=1).drop("Embarked", axis=1)

    return df

def normalize_for_nets(df: pd.DataFrame) -> pd.DataFrame:
    norm_df = (df - df.min()) / (df.max() - df.min())
    return norm_df

def separate_data_and_target(df: pd.DataFrame, targets: List[str]) -> (pd.DataFrame, pd.DataFrame):
    data = df.loc[:, df.columns != targets]
    target = df.loc[:, targets]
    return data, target

def clean_missing(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=["number"]).columns
    non_numeric_cols = df.select_dtypes(exclude=["number"]).columns

    median = df[numeric_cols].median()
    most_freq = df[non_numeric_cols].describe().loc['top']

    df[numeric_cols] = df[numeric_cols].fillna(median)
    df[non_numeric_cols] = df[non_numeric_cols].fillna(most_freq)

    return df

def clean_outliers(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Removes severe outliers
    changes mild outliers to lower or upper fence
    '''
    cols = df.select_dtypes(include=["number"]).columns
    if "Survived" in cols:
        cols = cols.drop(labels=["Survived"])

    # Extreme outliers
    Q1 = df[cols].quantile(0.05)
    Q3 = df[cols].quantile(0.95)
    IQR = Q3 - Q1
    lower_fences = Q1 - 1.5 * IQR
    upper_fences = Q3 + 1.5 * IQR
    is_outlier = ~((df[cols] < lower_fences) | (df[cols] > upper_fences)).any(axis=1)

    filtered_df = df[is_outlier].copy()

    #Mild outliers
    Q1 = filtered_df[cols].quantile(0.25)
    Q3 = filtered_df[cols].quantile(0.75)
    IQR = Q3 - Q1
    lower_fences = Q1 - 1.5 * IQR
    upper_fences = Q3 + 1.5 * IQR
    upper_fences["Parch"] = 2
    for col in cols:
        filtered_df.loc[filtered_df[col] < lower_fences[col], col] = math.ceil(lower_fences[col])
        filtered_df.loc[filtered_df[col] > upper_fences[col], col] = math.floor(upper_fences[col])

    return filtered_df