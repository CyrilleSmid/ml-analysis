from typing import Tuple
import pandas as pd

import data_analysis

def clean_and_separate_by_type(df: pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame]:
    numeric_cols = df.select_dtypes(include=["number"]).columns

    non_numeric_cols = df.select_dtypes(exclude=["number"]).columns

    # num_missing = df.isna().sum()
    # print(num_missing)

    pct_missing = df.isna().mean()
    data_analysis.visualize_missing_data(df)