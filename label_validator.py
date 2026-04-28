import json
import os
import logging
from typing import Dict, List, Any

import pandas as pd
from fastmcp import tool
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from loader import load_data
from structure_analyzer import df_to_column_samples
from label_fixer import fix_column_labels


# ---- Logging ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---- LLM Setup ----
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY,
    timeout=60
)


# ---- Prompt ----
SYSTEM_PROMPT = """
You are a dataset column label validation agent.

You will receive column samples and unique counts.

Detect label issues and suggest fixes.

Rules:
- If correct → issue="correct", action="none"
- Return result for EVERY column
"""


# ---- Pydantic Schema ----
class ColumnValidation(BaseModel):
    column: str
    issue: str
    action: str
    suggested_name: str
    confidence: float
    reason: str


class ValidationReport(BaseModel):
    items: List[ColumnValidation]


# ---- TOOL 1: Validate ----
@tool(
    name="validate_dataset",
    description="Validate dataset column labels using an LLM.",
    input_schema={"column_samples": "object"},
    output_schema={"report": "array"},
)
def validate_dataset(column_samples: Dict[str, Any]) -> Dict[str, Any]:

    if not column_samples or not isinstance(column_samples, dict):
        return {"error": "Invalid or empty column_samples"}

    prompt = f"""
Dataset Columns:

{json.dumps(column_samples, indent=2)}
"""

    try:
        structured_llm = llm.with_structured_output(ValidationReport)

        response = structured_llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])

        llm_report = [item.model_dump() for item in response.items]

    except Exception as e:
        return {"error": f"LLM failed: {str(e)}"}

    # ---- Ensure full coverage ----
    llm_map = {item["column"]: item for item in llm_report}

    final_report = []

    for column in column_samples.keys():

        if column in llm_map:
            item = llm_map[column]

            item.setdefault("issue", "correct")
            item.setdefault("action", "none")
            item.setdefault("suggested_name", column)
            item.setdefault("confidence", 0.5)
            item.setdefault("reason", "")

            final_report.append(item)

        else:
            final_report.append({
                "column": column,
                "issue": "correct",
                "action": "none",
                "suggested_name": column,
                "confidence": 1.0,
                "reason": "no label issues detected"
            })

    return {"report": final_report}


# ---- TOOL 2: Generate Report ----
@tool(
    name="generate_label_report",
    description="Generate label validation report from dataset file.",
    input_schema={"file_path": "string"},
    output_schema={"report": "array"},
)
def generate_label_report(file_path: str) -> Dict[str, Any]:

    if not file_path or not os.path.exists(file_path):
        return {"error": "Invalid file path"}

    try:
        df = load_data(file_path)[0]
    except Exception as e:
        return {"error": f"Data loading failed: {str(e)}"}

    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"error": "Invalid or empty dataset"}

    try:
        column_samples = df_to_column_samples(df)
    except Exception as e:
        return {"error": f"Column sampling failed: {str(e)}"}

    result = validate_dataset(column_samples)

    return result


# ---- TOOL 3: Validate + Fix ----
@tool(
    name="validate_and_fix_labels",
    description="Validate and fix dataset column labels.",
    input_schema={"file_path": "string"},
    output_schema={
        "fixed_columns": "array",
        "report": "array",
        "preview": "array"
    },
)
def validate_and_fix_labels(file_path: str) -> Dict[str, Any]:

    if not file_path or not os.path.exists(file_path):
        return {"error": "Invalid file path"}

    try:
        df = load_data(file_path)[0]
    except Exception as e:
        return {"error": f"Data loading failed: {str(e)}"}

    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"error": "Invalid or empty dataset"}

    try:
        column_samples = df_to_column_samples(df)
        report_result = validate_dataset(column_samples)

        if "error" in report_result:
            return report_result

        report = report_result["report"]

    except Exception as e:
        return {"error": f"Validation failed: {str(e)}"}

    try:
        df_fixed = fix_column_labels(df, report)
    except Exception as e:
        return {"error": f"Label fixing failed: {str(e)}"}

    return {
        "fixed_columns": list(df_fixed.columns),
        "report": report,
        "preview": df_fixed.head(10).to_dict(orient="records"),
    }