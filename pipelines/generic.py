import pandas as pd


def drop_columns(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return data.drop(columns, axis=1)
