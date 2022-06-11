import pandas as pd

import data_analysis
import data_cleaning

if __name__ == "__main__":
    df = pd.read_csv(r"../datasets/train.csv", sep=",", header="infer", names=None, encoding="utf-8")
    pd.set_option("display.max_rows", None, "display.max_columns", None)

    # print(df.head(5))

    df.info()

    data_cleaning.clean_and_separate_by_type(df)
