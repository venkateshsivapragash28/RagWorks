import pandas as pd
from fastmcp import tool

def get_column_info(df):

    column_info = {}

    for column in df.columns:
        column_info[column] = {
            "dtype": str(df[column].dtype),
            "nunique": int(df[column].nunique())
        }

    return column_info