import pandas as pd

import data_analysis
import data_cleaning
import decision_tree

def predict_and_save(model, ids, X_test) -> None:
    pred_df = pd.DataFrame(ids)
    y_pred = decision_tree.predict(X_test)
    pred_df["Survived"] = pd.Series(y_pred)
    pred_df.to_csv(r"../datasets/decision_tree_submission.csv", index=False)

if __name__ == "__main__":
    df = pd.read_csv(r"../datasets/train.csv", sep=",", header="infer", names=None, encoding="utf-8")
    pd.set_option("display.max_rows", None, "display.max_columns", None)

    df = data_cleaning.clean_training_dataset(df)
    decision_tree = decision_tree.get_trained_model(df)

    test_df = pd.read_csv(r"../datasets/test.csv", sep=",", header="infer", names=None, encoding="utf-8")
    ids, test_df = data_cleaning.normalize_for_prediction(test_df)
    predict_and_save(decision_tree, ids, test_df)
