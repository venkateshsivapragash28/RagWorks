import pandas as pd
import os
from fastmcp import tool


def _load_dataframe(file_path: str):
    if not file_path or not isinstance(file_path, str):
        return None, "Invalid file_path"

    if not os.path.exists(file_path):
        return None, f"File not found: {file_path}"

    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file_path)
        else:
            return None, "Unsupported file format"

        return df, None

    except Exception as e:
        return None, str(e)


@tool(
    name="split_dataset",
    description="Split a dataset into train/test previews using a ratio.",
    input_schema={
        "file_path": "string",
        "test_ratio": "number",
        "random_state": "integer"
    },
    output_schema={
        "train_preview": "array",
        "test_preview": "array",
        "metadata": "object"
    },
)
def split_dataset(file_path: str, test_ratio: float = 0.2, random_state: int = 42):

    # ---- Validate ratio ----
    if not isinstance(test_ratio, (int, float)) or not (0 <= test_ratio <= 1):
        return {
            "train_preview": [],
            "test_preview": [],
            "metadata": {},
            "error": "test_ratio must be between 0 and 1"
        }

    # ---- Load data safely ----
    df, err = _load_dataframe(file_path)
    if err:
        return {
            "train_preview": [],
            "test_preview": [],
            "metadata": {},
            "error": err
        }

    if df.empty:
        return {
            "train_preview": [],
            "test_preview": [],
            "metadata": {
                "num_rows": 0,
                "num_columns": 0,
                "train_rows": 0,
                "test_rows": 0
            }
        }

    # ---- Split logic (unchanged) ----
    test_size = int(len(df) * test_ratio)

    train_df = df.iloc[:-test_size] if test_size > 0 else df
    test_df = df.iloc[-test_size:] if test_size > 0 else df.iloc[0:0]

    metadata = {
        "num_rows": int(len(df)),
        "num_columns": int(df.shape[1]),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "test_ratio": float(test_ratio)
    }

    return {
        "train_preview": train_df.head(10).to_dict(orient="records"),
        "test_preview": test_df.head(10).to_dict(orient="records"),
        "metadata": metadata
    }