import json
import sys
import os
import logging
from typing import Dict, List, Any, Tuple, Optional
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from loader import load_data
from structure_analyzer import df_to_column_samples
from label_fixer import fix_column_labels

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
import pandas as pd
from fastmcp import tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY is not set")
    raise ValueError("GOOGLE_API_KEY environment variable is required")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY,
    timeout=60
)


SYSTEM_PROMPT = """
You are a dataset column label validation agent.

You will receive a dictionary:

{
 "column_name": {"samples": [sample_values], "nunique": count},
 "column_name": {"samples": [sample_values], "nunique": count}
}

Your job is to detect ALL column label inconsistencies.

Detect these issue types:
missing_name, noise, mismatch, swap, duplicate, ambiguous,
inconsistent_style, reserved_name, label_typo, special_character,
missing_label, duplicate_label.

Rules:
- If column is correct → issue = "correct", action="none"
- Return a result for EVERY column
"""

class ColumnValidation(BaseModel):
    column: str = Field(description="The original column name")
    issue: str = Field(description="Issue type (e.g. missing_name, noise, etc.) or 'correct'")
    action: str = Field(description="'rename', 'review', or 'none'")
    suggested_name: str = Field(description="Suggested column name if renamed, otherwise original")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")
    reason: str = Field(description="Short explanation")

class ValidationReport(BaseModel):
    items: List[ColumnValidation] = Field(description="List of validation results for every column")


@tool(
    name="validate_dataset",
    description="Validate dataset column labels using an LLM-based label validation agent.",
    input_schema={"column_samples": "object"},
    output_schema={"report": "array"},
)
def validate_dataset(column_samples: Dict[str, Any]) -> List[Dict[str, Any]]:
    
    if not column_samples:
        logger.warning("Empty column_samples provided")
        return []
    
    if not isinstance(column_samples, dict):
        logger.error(f"Invalid column_samples type: {type(column_samples)}")
        raise TypeError("column_samples must be a dictionary")

    prompt = f"""
Dataset Columns and Sample Values:

{json.dumps(column_samples, indent=2)}

Return a validation result for EVERY column in the dataset.
"""

    try:
        structured_llm = llm.with_structured_output(ValidationReport)
        response = structured_llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])
        llm_report = [item.model_dump() for item in response.items] if response and response.items else []
    except Exception as e:
        logger.error(f"LLM API call failed: {str(e)}")
        raise RuntimeError(f"Failed to invoke LLM: {str(e)}") from e

    # Convert LLM output into dictionary
    llm_map = {item["column"]: item for item in llm_report}

    final_report = []

    for column in column_samples.keys():

        if column in llm_map:

            item = llm_map[column]

            # Ensure required fields exist
            item.setdefault("issue", "correct")
            item.setdefault("action", "none")
            item.setdefault("suggested_name", column)
            item.setdefault("confidence", 0.5)
            item.setdefault("reason", "")
            
            # Validate field types
            if not isinstance(item.get("confidence"), (int, float)):
                logger.warning(f"Invalid confidence for {column}, defaulting to 0.5")
                item["confidence"] = 0.5

            final_report.append(item)

        else:
            logger.debug(f"No LLM result for column '{column}', marking as correct")
            final_report.append({
                "column": column,
                "issue": "correct",
                "action": "none",
                "suggested_name": column,
                "confidence": 1.0,
                "reason": "no label issues detected"
            })

    logger.info(f"Validated {len(final_report)} columns")
    return final_report


@tool(
    name="generate_label_report",
    description="Generate a column label validation report for a dataset file.",
    input_schema={"file_path": "string"},
    output_schema={"report": "array"},
)
def generate_label_report(file_path: str) -> List[Dict[str, Any]]:
    
    if not file_path:
        raise ValueError("file_path cannot be empty")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        data_result = load_data(file_path)
        if not data_result or len(data_result) == 0:
            raise ValueError("load_data returned empty result")
        df = data_result[0]
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {str(e)}")
        raise RuntimeError(f"Data loading failed: {str(e)}") from e
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df)}")
    
    if df.empty:
        logger.warning("DataFrame is empty")
        return []

    try:
        column_samples = df_to_column_samples(df)
    except Exception as e:
        logger.error(f"Failed to extract column samples: {str(e)}")
        raise RuntimeError(f"Column sampling failed: {str(e)}") from e

    report = validate_dataset(column_samples)

    return report


@tool(
    name="validate_and_fix_labels",
    description="Validate and fix dataset column labels from a file.",
    input_schema={"file_path": "string"},
    output_schema={"fixed_columns": "array", "report": "array", "preview": "array"},
)
def validate_and_fix_labels(file_path: str) -> Dict[str, Any]:
    
    if not file_path:
        raise ValueError("file_path cannot be empty")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        data_result = load_data(file_path)
        if not data_result or len(data_result) == 0:
            raise ValueError("load_data returned empty result")
        df = data_result[0]
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {str(e)}")
        raise RuntimeError(f"Data loading failed: {str(e)}") from e
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df)}")
    
    if df.empty:
        logger.warning("DataFrame is empty")
        return {
            "fixed_columns": [],
            "report": [],
            "preview": [],
        }

    try:
        column_samples = df_to_column_samples(df)
    except Exception as e:
        logger.error(f"Failed to extract column samples: {str(e)}")
        raise RuntimeError(f"Column sampling failed: {str(e)}") from e

    report = validate_dataset(column_samples)

    try:
        df_fixed = fix_column_labels(df, report)
    except Exception as e:
        logger.error(f"Failed to fix column labels: {str(e)}")
        raise RuntimeError(f"Label fixing failed: {str(e)}") from e

    return {
        "fixed_columns": list(df_fixed.columns),
        "report": report,
        "preview": df_fixed.head(10).to_dict(orient="records"),
    }


if __name__ == "__main__":

    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Automobile_data.csv")
    
    if not os.path.exists(file_path):
        logger.error(f"Test file not found: {file_path}")
        sys.exit(1)

    try:
        df_fixed, report = validate_and_fix_labels(file_path)
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        sys.exit(1)

    print("\n===== COLUMN LABEL INCONSISTENCY REPORT =====\n")

    for r in report:

        print(f"column : {r['column']}")
        print(f"issue : {r['issue']}")
        print(f"suggested name : {r['suggested_name']}")
        print(f"action : {r['action']}")
        print(f"confidence : {r['confidence']}")
        print(f"reason : {r['reason']}\n")

    try:
        original_df = load_data(file_path)[0]
        print("Original columns:", list(original_df.columns))
        print("Fixed columns:", list(df_fixed.columns))
    except Exception as e:
        logger.error(f"Failed to display comparison: {str(e)}")
