import pandas as pd
from fastmcp import tool

def df_to_column_samples(df, sample_size=10):

    column_samples = {}

    for column in df.columns:

        samples = (
            df[column]
            .dropna()
            .astype(str)
            .sample(min(sample_size, len(df)))
            .tolist()
        )

        nunique = df[column].nunique()

        column_samples[column] = {
            "samples": samples,
            "nunique": nunique
        }

    return column_samples