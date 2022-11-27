import pandas as pd
from sklearn.model_selection import train_test_split

from pipelines.generic import drop_columns
from pipelines.hidden_nulls import (
    detect_hidden_nulls,
    replace_hidden_nulls,
    replace_nulls_in_numeric_columns,
    winsortize_outliers_in_columns,
    standardize_numeric_columns,
)

pd.set_option("display.max_columns", None)

if __name__ == "__main__":
    data = pd.read_csv("data/raw_data.csv", sep=";")
    train_data, test_data = train_test_split(data, test_size=0.2)
    # detect_hidden_nulls(data, data.columns)
    train_data = drop_columns(train_data, ["euribor3m", "emp.var.rate"])
    train_data_without_hidden_nulls = replace_hidden_nulls(
        train_data, train_data.columns.tolist(), custom_values={"pdays": [999]}
    )
    train_data_without_nulls = replace_nulls_in_numeric_columns(train_data)

    train_data_without_nulls.save("data/data_without_nulls.csv", index=False)

    train_data = winsortize_outliers_in_columns(
        train_data_without_nulls,
        ["campaign", "pdays", "cons.price.idx", "cons.conf.idx"],
        capping_method="gaussian",
    )
    train_data = standardize_numeric_columns(
        train_data,
    )
