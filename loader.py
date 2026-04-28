import pandas as pd
import os
from fastmcp import tool


def load_data(file_path):
    try:
        if file_path.endswith(".csv"):
            data = pd.read_csv(file_path)
        elif file_path.endswith((".xls", ".xlsx")):
            data = pd.read_excel(file_path)
        else:
            return None, {"error": "Unsupported file format"}

        metadata = {
            "num_rows": int(data.shape[0]),
            "num_columns": int(data.shape[1]),
            "column_names": data.columns.tolist(),
            "data_types": {col: str(dtype) for col, dtype in data.dtypes.items()},
        }

        return data, metadata

    except Exception as e:
        return None, {"error": str(e)}


@tool(
    name="load_data",
    description="Load a CSV or Excel dataset and return metadata plus preview rows.",
    input_schema={"file_path": "string"},
    output_schema={"metadata": "object", "preview": "array"},
)
def load_data_tool(file_path: str):

    # ---- Validate input ----
    if not file_path or not isinstance(file_path, str):
        return {"error": "Invalid file_path"}

    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}

    # ---- Load data ----
    df, metadata = load_data(file_path)

    if df is None:
        return {"error": metadata.get("error", "Unknown loading error")}

    # ---- Create preview ----
    preview = df.head(10).to_dict(orient="records")

    return {
        "metadata": metadata,
        "preview": preview,
    }