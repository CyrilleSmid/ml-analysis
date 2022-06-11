from typing import Tuple
import pandas as pd

import data_analysis

def clean_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    df_clean_norm = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

    df_clean_norm = clean_missing(df_clean_norm)

    df_clean_norm = clean_outliers(df_clean_norm)

    return df_clean_norm

def clean_missing(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=["number"]).columns
    non_numeric_cols = df.select_dtypes(exclude=["number"]).columns

    median = df[numeric_cols].median()
    most_freq = df[non_numeric_cols].describe().loc['top']

    df[numeric_cols] = df[numeric_cols].fillna(median)
    df[non_numeric_cols] = df[non_numeric_cols].fillna(most_freq)

    return df

def clean_outliers(df: pd.DataFrame) -> pd.DataFrame:

    return df