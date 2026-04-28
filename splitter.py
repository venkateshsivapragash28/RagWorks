import pandas as pd
from fastmcp import tool


def _load_dataframe(file_path: str) -> pd.DataFrame:
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith((".xls", ".xlsx")):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file extension for split_dataset")


@tool(
    name="split_dataset",
    description="Split a dataset file into train/test previews using a requested ratio.",
    input_schema={"file_path": "string", "test_ratio": "number", "random_state": "integer"},
    output_schema={"train_preview": "array", "test_preview": "array", "metadata": "object"},
)
def split_dataset(file_path: str, test_ratio: float = 0.2, random_state: int = 42):
    df = _load_dataframe(file_path)
    if df.empty:
        return {"train_preview": [], "test_preview": [], "metadata": {"num_rows": 0, "num_columns": 0}}

    test_size = int(len(df) * test_ratio)
    train_df = df.iloc[:-test_size] if test_size > 0 else df
    test_df = df.iloc[-test_size:] if test_size > 0 else df.iloc[0:0]

    metadata = {
        "num_rows": len(df),
        "num_columns": df.shape[1],
        "train_rows": len(train_df),
        "test_rows": len(test_df),
    }

    return {
        "train_preview": train_df.head(10).to_dict(orient="records"),
        "test_preview": test_df.head(10).to_dict(orient="records"),
        "metadata": metadata,
    }
