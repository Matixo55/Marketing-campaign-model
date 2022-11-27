from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer

from utils import NULL_EQUIVALENTS

NULL_EQUIVALENTS = NULL_EQUIVALENTS + [0]


def detect_hidden_nulls(data: pd.DataFrame, columns: list[str]) -> None:
    print("Potential hidden nulls:")

    for column in columns:
        print(f"\nColumn: {column}")

        for value in NULL_EQUIVALENTS:
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
    strategies: dict[str, str] = None,
    fill_values: dict[str, str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    :param data:
    :param strategies: Dictionary with one of the following strategies: "mean", "median", "most_frequent", "constant"]
    :param fill_values: Dictionary with replacement values for "constant" strategy
    :param kwargs: arguments for Imputer
    """
    numeric_columns = data.select_dtypes(include=[np.number]).columns.values

    for column in numeric_columns:
        imputer_kwargs = {}
        if strategies and column in strategies:
            imputer_kwargs["strategy"] = strategies[column]
        if fill_values and column in fill_values:
            imputer_kwargs["fill_value"] = fill_values[column]

        data = replace_nulls_in_numeric_column(data, column, **imputer_kwargs, **kwargs)

    return data


def replace_nulls_in_numeric_column(
    data: pd.DataFrame,
    column: str,
    strategy: str = "constant",
    **kwargs,
) -> pd.DataFrame:
    """
    :param data:
    :param column: Column name
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

    data[column] = imputer.fit_transform(data[[column]])

    return data
