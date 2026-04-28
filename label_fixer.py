import pandas as pd
from fastmcp import tool

def fix_column_labels(df, report):
    """
    Fix column label inconsistencies in the DataFrame based on the validation report.

    Args:
        df (pd.DataFrame): The original DataFrame
        report (list): List of validation report items from label_validator

    Returns:
        pd.DataFrame: DataFrame with fixed column names
    """
    rename_dict = {}

    for item in report:
        if item.get("action") == "rename":
            original = item["column"]
            suggested = item["suggested_name"]
            if original in df.columns:
                rename_dict[original] = suggested

    df_fixed = df.rename(columns=rename_dict)
    return df_fixed


@tool(
    name="fix_column_labels",
    description="Fix dataset column labels based on a label validation report.",
    input_schema={"file_path": "string", "report": "array"},
    output_schema={"fixed_columns": "array", "preview": "array"},
)
def fix_column_labels_tool(file_path: str, report):
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file extension for fix_column_labels_tool")

    df_fixed = fix_column_labels(df, report)
    return {
        "fixed_columns": df_fixed.columns.tolist(),
        "preview": df_fixed.head(10).to_dict(orient="records"),
    }