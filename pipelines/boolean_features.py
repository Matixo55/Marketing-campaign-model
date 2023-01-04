import pandas as pd


def replace_obscured_booleans(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        data[column] = data[column].replace(["yes", "no"], [1, 0])
        data[column] = data[column].replace(["Yes", "No"], [1, 0])

    return data
