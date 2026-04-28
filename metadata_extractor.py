import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
import re
from fastmcp import tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_column_metadata(df: pd.DataFrame, column: str, sample_size: int = 100) -> Optional[Dict[str, Any]]:
    """Extract comprehensive metadata for a single column."""
    if column not in df.columns:
        logger.error(f"Column '{column}' not found in DataFrame")
        return None
    
    try:
        col_data = df[column]
        total_count = len(col_data)
        missing_count = col_data.isna().sum()
        non_missing = col_data.dropna()
        
        if len(non_missing) == 0:
            logger.warning(f"Column '{column}' has no non-null values")
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
                "has_mixed_types": False
            }
        
        sample_count = min(sample_size, len(non_missing))
        samples = non_missing.sample(n=sample_count, random_state=42).astype(str).tolist()
        
        numeric_count = 0
        for val in non_missing:
            try:
                if pd.notna(val):
                    str_val = str(val).strip()
                    if re.search(r'\d', str_val):
                        numeric_count += 1
            except:
                continue
        
        numeric_ratio = numeric_count / len(non_missing) if len(non_missing) > 0 else 0.0
        
        has_units = False
        if col_data.dtype == 'object' and numeric_ratio > 0.5:
            unit_pattern = r'^\s*[\$\€\£\₹]?\s*\d+\.?\d*\s*[a-zA-Z%]+\s*$'
            unit_samples = [s for s in samples if re.match(unit_pattern, str(s))]
            has_units = len(unit_samples) > len(samples) * 0.3
        
        type_set = set()
        for val in non_missing.head(50):
            type_set.add(type(val).__name__)
        has_mixed_types = len(type_set) > 1
        
        metadata = {
            "column": column,
            "dtype": str(col_data.dtype),
            "total_count": int(total_count),
            "missing_count": int(missing_count),
            "missing_ratio": float(missing_count / total_count) if total_count > 0 else 0.0,
            "nunique": int(col_data.nunique()),
            "samples": samples[:20],
            "numeric_ratio": float(numeric_ratio),
            "has_units": bool(has_units),
            "has_mixed_types": bool(has_mixed_types)
        }
        
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to extract metadata for column '{column}': {str(e)}")
        return None


def extract_dataset_metadata(df: pd.DataFrame, sample_size: int = 100) -> Dict[str, Dict[str, Any]]:
    """Extract metadata for all columns in a DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df)}")
    
    if df.empty:
        logger.warning("DataFrame is empty")
        return {}
    
    metadata = {}
    
    for column in df.columns:
        col_metadata = extract_column_metadata(df, column, sample_size)
        if col_metadata:
            metadata[column] = col_metadata
        else:
            logger.warning(f"Skipping column '{column}' due to extraction failure")
    
    logger.info(f"Extracted metadata for {len(metadata)} columns")
    return metadata


@tool(
    name="extract_column_metadata",
    description="Extract metadata for a specific column in a dataset.",
    input_schema={"file_path": "string", "column": "string", "sample_size": "integer"},
    output_schema={"column_metadata": "object"},
)
def extract_column_metadata_tool(file_path: str, column: str, sample_size: int = 100):
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file extension for extract_column_metadata_tool")
    return {"column_metadata": extract_column_metadata(df, column, sample_size)}


@tool(
    name="extract_dataset_metadata",
    description="Extract metadata for all columns in a dataset.",
    input_schema={"file_path": "string", "sample_size": "integer"},
    output_schema={"metadata": "object"},
)
def extract_dataset_metadata_tool(file_path: str, sample_size: int = 100):
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file extension for extract_dataset_metadata_tool")
    return {"metadata": extract_dataset_metadata(df, sample_size=sample_size)}


@tool(
    name="identify_semantic_candidates",
    description="Identify semantic normalization candidate columns from metadata.",
    input_schema={"metadata": "object", "numeric_ratio_threshold": "number"},
    output_schema={"candidates": "object"},
)
def identify_semantic_candidates_tool(metadata: Dict[str, Dict[str, Any]], numeric_ratio_threshold: float = 0.7):
    return {"candidates": identify_semantic_candidates(metadata, numeric_ratio_threshold)}


def identify_semantic_candidates(metadata: Dict[str, Dict[str, Any]], 
                                 numeric_ratio_threshold: float = 0.7) -> Dict[str, Dict[str, Any]]:
    """
    Identify columns that are candidates for semantic normalization.
    
    Candidates are columns that:
    - Have object dtype
    - Have high numeric ratio (contain numbers)
    - Likely have units or formatting issues
    """
    if not metadata:
        logger.warning("Empty metadata provided")
        return {}
    
    candidates = {}
    
    for col_name, col_meta in metadata.items():
        is_object = col_meta.get("dtype") == "object"
        numeric_ratio = col_meta.get("numeric_ratio", 0.0)
        has_units = col_meta.get("has_units", False)
        
        if is_object and (numeric_ratio >= numeric_ratio_threshold or has_units):
            candidates[col_name] = col_meta
            logger.debug(f"Column '{col_name}' identified as semantic candidate "
                        f"(numeric_ratio={numeric_ratio:.2f}, has_units={has_units})")
    
    logger.info(f"Identified {len(candidates)} semantic normalization candidates")
    return candidates
