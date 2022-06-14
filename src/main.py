import pandas as pd

import data_analysis
import data_cleaning
import decision_tree

if __name__ == "__main__":
    df = pd.read_csv(r"../datasets/train.csv", sep=",", header="infer", names=None, encoding="utf-8")
    pd.set_option("display.max_rows", None, "display.max_columns", None)

    df = data_cleaning.clean(df)
    decision_tree = decision_tree.get_trained_model(df)

    test_df = pd.read_csv(r"../datasets/test.csv", sep=",", header="infer", names=None, encoding="utf-8")
    pred_df = pd.DataFrame(test_df["PassengerId"])
    test_df = test_df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
    test_df = data_cleaning.clean_missing(test_df)
    test_df = data_cleaning.notmalize_categorical(test_df)

    y_pred = decision_tree.predict(test_df)
    pred_df["Survived"] = pd.Series(y_pred)
    print(pred_df)
    pred_df.to_csv(r"../datasets/decision_tree_submission.csv", index=False)