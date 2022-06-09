import pandas as pd

import data_analysis

if __name__ == "__main__":
    dataset = pd.read_csv(r"../datasets/train.csv", sep=",", header="infer", names=None, encoding="utf-8")
    pd.set_option("display.max_rows", None, "display.max_columns", None)

    print(dataset.head(5))

    data_analysis.visualize(dataset)


    

