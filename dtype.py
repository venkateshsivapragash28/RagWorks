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


@tool(
    name="get_column_info",
    description="Return dtype and unique value count for each column in a dataset.",
    input_schema={"file_path": "string"},
    output_schema={"columns": "object"},
)
def get_column_info_tool(file_path: str):

    # Load dataset
    df = pd.read_csv(file_path)

    # Use your original logic
    column_info = get_column_info(df)

    return {"columns": column_info}



from fastmcp import FastMCP

mcp = FastMCP("column-info")

if __name__ == "__main__":
    mcp.run()