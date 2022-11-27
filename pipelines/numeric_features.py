from feature_engine.outliers import Winsorizer, OutlierTrimmer
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


def winsortize_outliers_in_columns(
    data: pd.DataFrame, columns: list[str], capping_method: str = "iqr", **kwargs
) -> pd.DataFrame:
    """
    :param data:
    :param columns: columns to winsortize
    :param capping_method: one of ["gaussian", "iqr", "quantiles", "mad"]
    :param kwargs: arguments for Winsorizer
    """
    winsortizer = Winsorizer(capping_method=capping_method, **kwargs)
    data[columns] = winsortizer.fit_transform(data[columns])

    return data


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


def standardize_numeric_columns(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    :param data:
    :param kwargs: arguments for StandardScaler
    """
    numeric_columns = data.select_dtypes(include=[np.number]).columns.values

    scaler = StandardScaler(**kwargs)
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    return data
