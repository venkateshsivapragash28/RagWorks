import pandas as pd
import logging
import os
import re
from typing import Dict, Any, Optional

from fastmcp import tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------
# SAFE DATA LOADER (REUSABLE)
# -----------------------------
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

        if df.empty:
            return None, "Dataset is empty"

        return df, None

    except Exception as e:
        return None, str(e)


# -----------------------------
# CORE LOGIC (UNCHANGED)
# -----------------------------
def extract_column_metadata(
    df: pd.DataFrame, column: str, sample_size: int = 100
) -> Optional[Dict[str, Any]]:

    if column not in df.columns:
        return None

    try:
        col_data = df[column]
        total_count = len(col_data)
        missing_count = col_data.isna().sum()
        non_missing = col_data.dropna()

        if len(non_missing) == 0:
            return {
                "column": column,
                "dtype": str(col_data.dtype),
                "total_count": total_count,
                "missing_count": missing_count,
                "missing_ratio": 1.0,
                "nunique": 0,
                "samples": [],
                "numeric_ratio": 0.0,
                "has_units": False,
                "has_mixed_types": False,
            }

        sample_count = min(sample_size, len(non_missing))
        samples = (
            non_missing.sample(n=sample_count, random_state=42)
            .astype(str)
            .tolist()
        )

        numeric_count = sum(
            1 for val in non_missing if re.search(r"\d", str(val))
        )

        numeric_ratio = numeric_count / len(non_missing)

        has_units = False
        if col_data.dtype == "object" and numeric_ratio > 0.5:
            unit_pattern = r"^\s*[\$\€\£\₹]?\s*\d+\.?\d*\s*[a-zA-Z%]+\s*$"
            unit_samples = [s for s in samples if re.match(unit_pattern, s)]
            has_units = len(unit_samples) > len(samples) * 0.3

        type_set = {type(val).__name__ for val in non_missing.head(50)}
        has_mixed_types = len(type_set) > 1

        return {
            "column": column,
            "dtype": str(col_data.dtype),
            "total_count": int(total_count),
            "missing_count": int(missing_count),
            "missing_ratio": float(missing_count / total_count)
            if total_count > 0
            else 0.0,
            "nunique": int(col_data.nunique()),
            "samples": samples[:20],
            "numeric_ratio": float(numeric_ratio),
            "has_units": bool(has_units),
            "has_mixed_types": bool(has_mixed_types),
        }

    except Exception:
        return None


def extract_dataset_metadata(
    df: pd.DataFrame, sample_size: int = 100
) -> Dict[str, Dict[str, Any]]:

    metadata = {}

    for column in df.columns:
        col_meta = extract_column_metadata(df, column, sample_size)
        if col_meta:
            metadata[column] = col_meta

    return metadata


def identify_semantic_candidates(
    metadata: Dict[str, Dict[str, Any]],
    numeric_ratio_threshold: float = 0.7,
) -> Dict[str, Dict[str, Any]]:

    candidates = {}

    for col_name, col_meta in metadata.items():
        if (
            col_meta.get("dtype") == "object"
            and (
                col_meta.get("numeric_ratio", 0.0)
                >= numeric_ratio_threshold
                or col_meta.get("has_units", False)
            )
        ):
            candidates[col_name] = col_meta

    return candidates


# -----------------------------
# MCP TOOLS (PRODUCTION SAFE)
# -----------------------------

@tool(
    name="extract_column_metadata",
    description="Extract metadata for a specific column.",
    input_schema={
        "file_path": "string",
        "column": "string",
        "sample_size": "integer",
    },
    output_schema={"column_metadata": "object"},
)
def extract_column_metadata_tool(
    file_path: str, column: str, sample_size: int = 100
):

    df, err = _load_dataframe(file_path)
    if err:
        return {"error": err}

    result = extract_column_metadata(df, column, sample_size)

    if result is None:
        return {"error": f"Column not found: {column}"}

    return {"column_metadata": result}


@tool(
    name="extract_dataset_metadata",
    description="Extract metadata for all columns in dataset.",
    input_schema={"file_path": "string", "sample_size": "integer"},
    output_schema={"metadata": "object"},
)
def extract_dataset_metadata_tool(
    file_path: str, sample_size: int = 100
):

    df, err = _load_dataframe(file_path)
    if err:
        return {"error": err}

    metadata = extract_dataset_metadata(df, sample_size)

    return {"metadata": metadata}


@tool(
    name="identify_semantic_candidates",
    description="Identify columns needing semantic normalization.",
    input_schema={
        "metadata": "object",
        "numeric_ratio_threshold": "number",
    },
    output_schema={"candidates": "object"},
)
def identify_semantic_candidates_tool(
    metadata: Dict[str, Dict[str, Any]],
    numeric_ratio_threshold: float = 0.7,
):

    if not metadata:
        return {"candidates": {}}

    candidates = identify_semantic_candidates(
        metadata, numeric_ratio_threshold
    )

    return {"candidates": candidates}