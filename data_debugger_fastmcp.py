from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from fastmcp import tool
from dtype import get_column_info
from loader import load_data
from metadata_extractor import extract_column_metadata, extract_dataset_metadata
from pattern_extractor import PatternExtractor
from statistical_guardrails import StatisticalGuardrails
from vectorized_cleaner import VectorizedCleaner


def _load_dataframe(file_path: str) -> pd.DataFrame:
    if not file_path or not isinstance(file_path, str):
        raise ValueError("file_path must be a non-empty string")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    result = load_data(file_path)
    if result is None:
        raise RuntimeError(f"load_data returned no result for {file_path}")

    if isinstance(result, tuple):
        df = result[0]
    else:
        df = result

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Loaded result is not a DataFrame: {type(df)}")

    return df


@tool(
    name="load_dataset",
    description="Load a CSV/XLSX dataset and return the first rows together with metadata.",
    input_schema={"file_path": "string"},
    output_schema={"metadata": "object", "preview": "array"},
)
def load_dataset(file_path: str) -> Dict[str, Any]:
    """Load a tabular dataset and return metadata plus preview rows."""
    df = _load_dataframe(file_path)
    result = load_data(file_path)
    data, metadata = result if isinstance(result, tuple) else (df, {})
    preview = df.head(10).to_dict(orient="records")
    return {"metadata": metadata, "preview": preview}


@tool(
    name="get_column_schema",
    description="Infer column data types and unique counts for the dataset.",
    input_schema={"file_path": "string"},
    output_schema={"columns": "object"},
)
def get_column_schema(file_path: str) -> Dict[str, Any]:
    """Return a simple schema of column dtypes and unique value counts."""
    df = _load_dataframe(file_path)
    return {"columns": get_column_info(df)}


@tool(
    name="extract_dataset_metadata",
    description="Extract metadata for all columns in the dataset, including missing ratios and sample values.",
    input_schema={"file_path": "string", "sample_size": "integer"},
    output_schema={"column_metadata": "object"},
)
def extract_dataset_metadata_tool(file_path: str, sample_size: int = 100) -> Dict[str, Any]:
    """Extract column metadata for the entire dataset."""
    df = _load_dataframe(file_path)
    metadata = extract_dataset_metadata(df, sample_size=sample_size)
    return {"column_metadata": metadata}


@tool(
    name="discover_column_patterns",
    description="Detect numeric formatting patterns and units in a column using regex extraction.",
    input_schema={"file_path": "string", "column": "string"},
    output_schema={"patterns": "object", "summary": "object"},
)
def discover_column_patterns(file_path: str, column: str) -> Dict[str, Any]:
    """Identify patterns, prefixes, suffixes, and unparseable values in a dataset column."""
    df = _load_dataframe(file_path)
    if column not in df.columns:
        raise ValueError(f"Column not found: {column}")

    extractor = PatternExtractor()
    series = df[column].astype(str)
    patterns = extractor.extract_column_patterns(series)
    summary = extractor.discover_unique_patterns(patterns)
    return {
        "patterns": patterns.to_dict(orient="records") if not patterns.empty else [],
        "summary": summary,
    }


@tool(
    name="validate_statistical_guardrails",
    description="Validate a cleaned numeric column against semantic expectations and anomaly guardrails.",
    input_schema={"file_path": "string", "column": "string", "semantic_type": "string", "z_score_threshold": "number"},
    output_schema={"validation_report": "object"},
)
def validate_statistical_guardrails_tool(
    file_path: str,
    column: str,
    semantic_type: str,
    z_score_threshold: float = 3.0,
) -> Dict[str, Any]:
    """Validate numeric distribution, range, outliers, and conversion sanity for one column."""
    df = _load_dataframe(file_path)
    if column not in df.columns:
        raise ValueError(f"Column not found: {column}")

    cleaner = VectorizedCleaner()
    guardrails = StatisticalGuardrails(z_score_threshold=z_score_threshold)

    original_series = df[column]
    cleaned_series = cleaner._extract_number_vectorized(original_series)
    report = guardrails.validate_column(
        original_series=original_series,
        cleaned_series=cleaned_series,
        column_name=column,
        semantic_type=semantic_type,
    )
    return {"validation_report": report}


@tool(
    name="clean_column_by_strategy",
    description="Clean one column using a semantic fix strategy and optional multiplier map.",
    input_schema={"file_path": "string", "column": "string", "fix_strategy": "string", "multipliers": "object", "standard_unit": "string"},
    output_schema={"cleaned_values": "array"},
)
def clean_column_by_strategy(
    file_path: str,
    column: str,
    fix_strategy: str,
    multipliers: Optional[Dict[str, float]] = None,
    standard_unit: str = "standard",
) -> Dict[str, Any]:
    """Clean a column using a vectorized strategy based on inferred semantic normalization."""
    df = _load_dataframe(file_path)
    if column not in df.columns:
        raise ValueError(f"Column not found: {column}")

    semantic_info = {
        "fix_strategy": fix_strategy,
        "multipliers": multipliers or {},
        "standard_unit": standard_unit,
    }
    cleaner = VectorizedCleaner()
    cleaned_series = cleaner.clean_column(df[column], semantic_info)
    return {"cleaned_values": cleaned_series.tolist()}
