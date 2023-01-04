import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


def encode_categorical_columns(
    data: pd.DataFrame,
    test_data: pd.DataFrame,
    columns: list[str] | str,
    strategy: str = "onehot",
    categories: list[str] = list,
    **kwargs,
) -> tuple[DataFrame, DataFrame]:
    """
    :param data:
    :param test_data:
    :param columns: List of columns to encode
    :param strategy: one of ["onehot", "ordinal", "trigonometric"]
    :param categories: list of categories to encode
    :param kwargs:  arguments for encoder
    :return:
    """
    if strategy == "onehot":
        encoder = OneHotEncoder(categories=categories, **kwargs)

        data[[columns]] = encoder.fit_transform(data[[columns]])
        test_data[[columns]] = encoder.transform(test_data[[columns]])
    elif strategy == "ordinal":
        encoder = OrdinalEncoder(categories=[categories], **kwargs)

        data[[columns]] = encoder.fit_transform(data[[columns]])
        test_data[[columns]] = encoder.transform(test_data[[columns]])
    elif strategy == "trigonometric":
        for column in columns:
            data[f"{column}_sin"] = np.sin(data[column])
            data[f"{column}_cos"] = np.cos(data[column])
            data = data.drop(columns=[column])

            test_data[f"{column}_sin"] = np.sin(test_data[column])
            test_data[f"{column}_cos"] = np.cos(test_data[column])
            test_data = test_data.drop(columns=[column])
    else:
        raise ValueError("Invalid encoding")

    return data, test_data
