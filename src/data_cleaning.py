from typing import Tuple
import pandas as pd
import math

import data_analysis

def clean_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

    df = clean_missing(df)
    df = clean_outliers(df)

    return df

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
