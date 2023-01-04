import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.compose import ColumnTransformer, make_column_transformer
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
        encoder = OneHotEncoder(sparse=False,**kwargs)
        encoded_data = encoder.fit_transform(data[columns])
        encoded_test_data = encoder.transform(test_data[columns])

        encoded_data = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out())
        encoded_test_data = pd.DataFrame(encoded_test_data, columns=encoder.get_feature_names_out())

        encoded_data.index = data.index
        encoded_test_data.index = test_data.index

        data = data.drop(columns, axis=1)
        test_data = test_data.drop(columns, axis=1)

        data = pd.concat([data, encoded_data], axis=1)
        test_data = pd.concat([test_data, encoded_test_data], axis=1)

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
