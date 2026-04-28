import pandas as pd
from fastmcp import tool


def fix_column_labels(df, report):
    """
    Fix column label inconsistencies in the DataFrame based on the validation report.
    """
    rename_dict = {}

    for item in report:
        if isinstance(item, dict) and item.get("action") == "rename":
            original = item.get("column")
            suggested = item.get("suggested_name")

            if original in df.columns and suggested:
                rename_dict[original] = suggested

    df_fixed = df.rename(columns=rename_dict)
    return df_fixed, rename_dict


@tool(
    name="fix_column_labels",
    description="Fix dataset column labels based on a label validation report.",
    input_schema={"file_path": "string", "report": "array"},
    output_schema={
        "fixed_columns": "array",
        "applied_changes": "object",
        "preview": "array"
    },
)
def fix_column_labels_tool(file_path: str, report):

    # ---- Validate input ----
    if not file_path or not isinstance(file_path, str):
        return {"error": "Invalid file_path provided"}

    if not isinstance(report, list):
        return {"error": "Report must be a list of validation items"}

    # ---- Load dataset safely ----
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file_path)
        else:
            return {"error": "Unsupported file format"}
    except Exception as e:
        return {"error": f"Failed to load dataset: {str(e)}"}

    # ---- Apply fix ----
    try:
        df_fixed, rename_dict = fix_column_labels(df, report)
    except Exception as e:
        return {"error": f"Failed to fix column labels: {str(e)}"}

    # ---- Return structured output ----
    return {
        "fixed_columns": df_fixed.columns.tolist(),
        "applied_changes": rename_dict,
        "preview": df_fixed.head(10).to_dict(orient="records"),
    }