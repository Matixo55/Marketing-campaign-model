from typing import Optional

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.impute import KNNImputer, SimpleImputer

from utils import NULL_EQUIVALENTS


def detect_hidden_nulls(data: pd.DataFrame, columns: list[str]) -> None:
    print("Potential hidden nulls:")

    for column in columns:
        print(f"\nColumn: {column}")

        for value in NULL_EQUIVALENTS + [0]:
            if (number_of_occurrences := data[data[column] == value].shape[0]) > 0:
                print(
                    f"Found {number_of_occurrences} [{round(number_of_occurrences * 100 / data.shape[0], 2)}%] of {value}"
                )


def replace_hidden_nulls(
    data: pd.DataFrame,
    columns: list[str],
    custom_values: Optional[dict[str, list]] = None,
    auto: bool = True,
) -> pd.DataFrame:
    """
    :param data:
    :param columns:
    :param custom_values: list of additional values to replace
    :param auto: replace all values from NULL_EQUIVALENTS
    """
    for column in columns:

        values_to_replace = (
            custom_values[column] if custom_values and column in custom_values else []
        )

        if auto:
            values_to_replace += NULL_EQUIVALENTS

        data[column] = data[column].replace(values_to_replace, np.NaN)

    return data


def replace_nulls_in_numeric_columns(
    data: pd.DataFrame,
    test_data: pd.DataFrame,
    columns: list[str],
    strategy: str = "constant",
    **kwargs,
) -> tuple[DataFrame, DataFrame]:
    """
    :param data:
    :param test_data:
    :param columns: Column names in which to replace nulls
    :param strategy: one of ["constant", "mean", "median", "KNN"]
    :param kwargs: arguments for Imputer
    """
    if strategy != "KNN":
        imputer = SimpleImputer(strategy=strategy, **kwargs)
    elif strategy == "KNN":
        imputer_params = {
            "n_neighbors": 5,
            "weights": "uniform",
            "missing_values": np.nan,
        } | kwargs

        imputer = KNNImputer(**imputer_params)
    else:
        raise ValueError("Invalid strategy")

    data[columns] = imputer.fit_transform(data[columns])
    test_data[columns] = imputer.transform(test_data[columns])

    return data, test_data


def replace_nulls_in_categorical_columns(
    data: pd.DataFrame,
    test_data: pd.DataFrame,
    columns: list[str],
    strategy: str = "most_frequent",
    constant: Optional[str] = None,
) -> tuple[DataFrame, DataFrame]:
    """
    :param data:
    :param test_data:
    :param columns: Column names in which to replace nulls
    :param strategy: one of ["constant", "most_frequent"]
    :param constant: value to replace nulls with
    """
    if strategy == "most_frequent":
        for column in columns:
            most_frequent_value = data[column].mode()[0]

            data[column] = data[column].fillna(most_frequent_value)
            test_data[column] = test_data[column].fillna(most_frequent_value)
    elif strategy == "constant":
        data[columns] = data[columns].fillna(constant)
        test_data[columns] = test_data[columns].fillna(constant)

    return data, test_data
