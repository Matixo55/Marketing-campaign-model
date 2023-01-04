from feature_engine.outliers import Winsorizer, OutlierTrimmer
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler


def winsortize_outliers_in_columns(
    data: pd.DataFrame,
    test_data: pd.DataFrame,
    columns: list[str],
    capping_method: str = "iqr",
    **kwargs,
) -> tuple[DataFrame, DataFrame]:
    """
    :param data:
    :param test_data:
    :param columns: columns to winsortize
    :param capping_method: one of ["gaussian", "iqr", "quantiles", "mad"]
    :param kwargs: arguments for Winsorizer
    """
    winsortizer = Winsorizer(capping_method=capping_method, **kwargs)
    data[columns] = winsortizer.fit_transform(data[columns])
    test_data[columns] = winsortizer.transform(test_data[columns])

    return data, test_data


def drop_columns_with_outliers(
    data: pd.DataFrame,
    excluded_columns: list[str],
    capping_method: str = "iqr",
    **kwargs,
) -> pd.DataFrame:
    """
    :param data:
    :param excluded_columns: columns to exclude from dropping
    :param capping_method: one of ["gaussian", "iqr", "quantiles", "mad"]
    :param kwargs: arguments for OutlierTrimmer
    """
    columns_to_drop = data.columns - excluded_columns
    outlier_trimmer = OutlierTrimmer(
        variables=columns_to_drop, capping_method=capping_method, **kwargs
    )
    data = outlier_trimmer.fit_transform(data)

    return data


def standardize_numeric_columns(
    data: pd.DataFrame, test_data: pd.DataFrame, columns: list[str], **kwargs
) -> tuple[DataFrame, DataFrame]:
    """
    :param data:
    :param test_data:
    :param kwargs: arguments for StandardScaler
    """
    scaler = StandardScaler(**kwargs)
    data[columns] = scaler.fit_transform(data[columns])
    test_data[columns] = scaler.transform(test_data[columns])

    return data, test_data
