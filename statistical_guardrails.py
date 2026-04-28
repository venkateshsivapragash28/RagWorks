import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from fastmcp import tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatisticalGuardrails:
    """Validates cleaning results and detects anomalies."""
    
    def __init__(self, z_score_threshold: float = 3.0):
        self.z_score_threshold = z_score_threshold
    
    def validate_range(self,
                      series: pd.Series,
                      column_name: str,
                      semantic_type: str,
                      min_val: Optional[float] = None,
                      max_val: Optional[float] = None) -> Dict[str, Any]:
        """
        Validate that values are within expected range.
        
        Args:
            series: Cleaned series to validate
            column_name: Name of the column
            semantic_type: Inferred semantic type
            min_val: Minimum expected value (optional)
            max_val: Maximum expected value (optional)
            
        Returns:
            Validation report
        """
        if series.empty or series.isna().all():
            return {
                "column": column_name,
                "check": "range_validation",
                "status": "skipped",
                "reason": "No valid values",
                "violations": []
            }
        
        numeric_series = pd.to_numeric(series, errors='coerce').dropna()
        
        if len(numeric_series) == 0:
            return {
                "column": column_name,
                "check": "range_validation",
                "status": "skipped",
                "reason": "No numeric values",
                "violations": []
            }
        
        # Auto-detect reasonable ranges based on semantic type
        if min_val is None or max_val is None:
            if semantic_type == "age":
                min_val = 0
                max_val = 120
            elif semantic_type == "percentage":
                min_val = 0
                max_val = 1
            elif semantic_type in ["mass", "distance", "storage", "duration"]:
                min_val = 0
                max_val = None
            elif semantic_type == "temperature":
                min_val = -273.15  # Absolute zero
                max_val = None
        
        violations = []
        
        if min_val is not None:
            below_min = numeric_series[numeric_series < min_val]
            if len(below_min) > 0:
                violations.append({
                    "type": "below_minimum",
                    "count": len(below_min),
                    "examples": below_min.head(5).tolist(),
                    "threshold": min_val
                })
        
        if max_val is not None:
            above_max = numeric_series[numeric_series > max_val]
            if len(above_max) > 0:
                violations.append({
                    "type": "above_maximum",
                    "count": len(above_max),
                    "examples": above_max.head(5).tolist(),
                    "threshold": max_val
                })
        
        status = "passed" if len(violations) == 0 else "failed"
        
        return {
            "column": column_name,
            "check": "range_validation",
            "status": status,
            "violations": violations,
            "total_values": len(numeric_series),
            "violation_ratio": sum(v["count"] for v in violations) / len(numeric_series) if violations else 0.0
        }
    
    def detect_outliers_zscore(self,
                               series: pd.Series,
                               column_name: str) -> Dict[str, Any]:
        """
        Detect outliers using Z-score method.
        
        Args:
            series: Series to check for outliers
            column_name: Name of the column
            
        Returns:
            Outlier detection report
        """
        if series.empty or series.isna().all():
            return {
                "column": column_name,
                "check": "zscore_outliers",
                "status": "skipped",
                "reason": "No valid values",
                "outliers": []
            }
        
        numeric_series = pd.to_numeric(series, errors='coerce').dropna()
        
        if len(numeric_series) < 3:
            return {
                "column": column_name,
                "check": "zscore_outliers",
                "status": "skipped",
                "reason": "Insufficient data points",
                "outliers": []
            }
        
        try:
            z_scores = np.abs(stats.zscore(numeric_series))
            outlier_mask = z_scores > self.z_score_threshold
            outliers = numeric_series[outlier_mask]
            
            if len(outliers) > 0:
                return {
                    "column": column_name,
                    "check": "zscore_outliers",
                    "status": "detected",
                    "outlier_count": len(outliers),
                    "outlier_ratio": len(outliers) / len(numeric_series),
                    "outlier_examples": outliers.head(10).tolist(),
                    "threshold": self.z_score_threshold
                }
            else:
                return {
                    "column": column_name,
                    "check": "zscore_outliers",
                    "status": "passed",
                    "outlier_count": 0,
                    "outlier_ratio": 0.0
                }
                
        except Exception as e:
            logger.error(f"Z-score outlier detection failed for '{column_name}': {str(e)}")
            return {
                "column": column_name,
                "check": "zscore_outliers",
                "status": "error",
                "reason": str(e)
            }
    
    def check_distribution_sanity(self,
                                  series: pd.Series,
                                  column_name: str,
                                  semantic_type: str) -> Dict[str, Any]:
        """
        Check if distribution makes sense for the semantic type.
        
        Args:
            series: Series to check
            column_name: Name of the column
            semantic_type: Inferred semantic type
            
        Returns:
            Distribution sanity report
        """
        if series.empty or series.isna().all():
            return {
                "column": column_name,
                "check": "distribution_sanity",
                "status": "skipped",
                "reason": "No valid values"
            }
        
        numeric_series = pd.to_numeric(series, errors='coerce').dropna()
        
        if len(numeric_series) < 10:
            return {
                "column": column_name,
                "check": "distribution_sanity",
                "status": "skipped",
                "reason": "Insufficient data points"
            }
        
        try:
            stats_dict = {
                "mean": float(numeric_series.mean()),
                "median": float(numeric_series.median()),
                "std": float(numeric_series.std()),
                "min": float(numeric_series.min()),
                "max": float(numeric_series.max()),
                "q25": float(numeric_series.quantile(0.25)),
                "q75": float(numeric_series.quantile(0.75))
            }
            
            warnings = []
            
            # Check for negative values in types that should be positive
            if semantic_type in ["age", "mass", "distance", "storage", "duration"]:
                negative_count = (numeric_series < 0).sum()
                if negative_count > 0:
                    warnings.append(f"Found {negative_count} negative values in {semantic_type} column")
            
            # Check for extreme skewness
            if stats_dict["std"] > 0:
                skewness = float(stats.skew(numeric_series))
                if abs(skewness) > 5:
                    warnings.append(f"Extreme skewness detected: {skewness:.2f}")
            
            # Check if all values are the same
            if stats_dict["std"] == 0:
                warnings.append("All values are identical")
            
            # Check for percentage values outside [0, 1]
            if semantic_type == "percentage":
                out_of_range = ((numeric_series < 0) | (numeric_series > 1)).sum()
                if out_of_range > 0:
                    warnings.append(f"{out_of_range} percentage values outside [0, 1] range")
            
            status = "passed" if len(warnings) == 0 else "warning"
            
            return {
                "column": column_name,
                "check": "distribution_sanity",
                "status": status,
                "statistics": stats_dict,
                "warnings": warnings
            }
            
        except Exception as e:
            logger.error(f"Distribution sanity check failed for '{column_name}': {str(e)}")
            return {
                "column": column_name,
                "check": "distribution_sanity",
                "status": "error",
                "reason": str(e)
            }
    
    def detect_pattern_mismatch(self,
                               original_series: pd.Series,
                               cleaned_series: pd.Series,
                               column_name: str,
                               semantic_type: str) -> Dict[str, Any]:
        """
        Detect if cleaned values don't match expected semantic type.
        
        Args:
            original_series: Original series before cleaning
            cleaned_series: Series after cleaning
            column_name: Name of the column
            semantic_type: Expected semantic type
            
        Returns:
            Pattern mismatch report
        """
        if original_series.empty or cleaned_series.empty:
            return {
                "column": column_name,
                "check": "pattern_mismatch",
                "status": "skipped",
                "reason": "Empty series"
            }
        
        # Check how many values became NaN after cleaning
        original_valid = original_series.notna().sum()
        cleaned_valid = pd.to_numeric(cleaned_series, errors='coerce').notna().sum()
        
        conversion_rate = cleaned_valid / original_valid if original_valid > 0 else 0.0
        
        warnings = []
        
        if conversion_rate < 0.5:
            warnings.append(f"Low conversion rate: {conversion_rate:.2%} of values successfully converted")
        
        if conversion_rate < 0.9:
            warnings.append(f"Some values failed to convert: {(1-conversion_rate):.2%} loss")
        
        status = "passed" if conversion_rate >= 0.9 else "warning" if conversion_rate >= 0.5 else "failed"
        
        return {
            "column": column_name,
            "check": "pattern_mismatch",
            "status": status,
            "conversion_rate": conversion_rate,
            "original_valid_count": int(original_valid),
            "cleaned_valid_count": int(cleaned_valid),
            "warnings": warnings
        }
    
    def validate_column(self,
                       original_series: pd.Series,
                       cleaned_series: pd.Series,
                       column_name: str,
                       semantic_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run all validation checks on a cleaned column.
        
        Args:
            original_series: Original series before cleaning
            cleaned_series: Series after cleaning
            column_name: Name of the column
            semantic_info: Semantic inference results
            
        Returns:
            Comprehensive validation report
        """
        semantic_type = semantic_info.get("semantic_type", "unknown")
        
        checks = []
        
        # Range validation
        range_check = self.validate_range(cleaned_series, column_name, semantic_type)
        checks.append(range_check)
        
        # Outlier detection
        outlier_check = self.detect_outliers_zscore(cleaned_series, column_name)
        checks.append(outlier_check)
        
        # Distribution sanity
        dist_check = self.check_distribution_sanity(cleaned_series, column_name, semantic_type)
        checks.append(dist_check)
        
        # Pattern mismatch
        mismatch_check = self.detect_pattern_mismatch(original_series, cleaned_series, column_name, semantic_type)
        checks.append(mismatch_check)
        
        # Overall status
        failed_checks = [c for c in checks if c.get("status") == "failed"]
        warning_checks = [c for c in checks if c.get("status") == "warning"]
        
        if failed_checks:
            overall_status = "failed"
        elif warning_checks:
            overall_status = "warning"
        else:
            overall_status = "passed"
        
        return {
            "column": column_name,
            "semantic_type": semantic_type,
            "overall_status": overall_status,
            "checks": checks,
            "failed_count": len(failed_checks),
            "warning_count": len(warning_checks)
        }
    
    def validate_dataframe(self,
                          original_df: pd.DataFrame,
                          cleaned_df: pd.DataFrame,
                          semantic_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate all cleaned columns in a DataFrame.
        
        Args:
            original_df: Original DataFrame before cleaning
            cleaned_df: DataFrame after cleaning
            semantic_results: Semantic inference results for all columns
            
        Returns:
            List of validation reports for each column
        """
        if not isinstance(original_df, pd.DataFrame) or not isinstance(cleaned_df, pd.DataFrame):
            raise TypeError("Both original_df and cleaned_df must be DataFrames")
        
        validation_reports = []
        
        for col_name, semantic_info in semantic_results.items():
            if col_name not in original_df.columns or col_name not in cleaned_df.columns:
                logger.warning(f"Column '{col_name}' not found in DataFrames")
                continue
            
            try:
                report = self.validate_column(
                    original_df[col_name],
                    cleaned_df[col_name],
                    col_name,
                    semantic_info
                )
                validation_reports.append(report)
                
            except Exception as e:
                logger.error(f"Validation failed for column '{col_name}': {str(e)}")
                continue
        
        logger.info(f"Validated {len(validation_reports)} columns")
        return validation_reports


@tool(
    name="validate_range",
    description="Validate numeric values against expected semantic ranges.",
    input_schema={"values": "array", "column_name": "string", "semantic_type": "string", "min_val": "number", "max_val": "number"},
    output_schema={"validation_report": "object"},
)
def validate_range_tool(values: List[Any], column_name: str, semantic_type: str, min_val: Optional[float] = None, max_val: Optional[float] = None):
    series = pd.Series(values)
    guardrails = StatisticalGuardrails()
    return {"validation_report": guardrails.validate_range(series, column_name, semantic_type, min_val, max_val)}


@tool(
    name="detect_outliers_zscore",
    description="Detect outliers using Z-score on a numeric value list.",
    input_schema={"values": "array", "column_name": "string", "z_score_threshold": "number"},
    output_schema={"outlier_report": "object"},
)
def detect_outliers_zscore_tool(values: List[Any], column_name: str, z_score_threshold: float = 3.0):
    series = pd.Series(values)
    guardrails = StatisticalGuardrails(z_score_threshold=z_score_threshold)
    return {"outlier_report": guardrails.detect_outliers_zscore(series, column_name)}


@tool(
    name="check_distribution_sanity",
    description="Check if a numeric distribution matches semantic expectations.",
    input_schema={"values": "array", "column_name": "string", "semantic_type": "string"},
    output_schema={"report": "object"},
)
def check_distribution_sanity_tool(values: List[Any], column_name: str, semantic_type: str):
    series = pd.Series(values)
    guardrails = StatisticalGuardrails()
    return {"report": guardrails.check_distribution_sanity(series, column_name, semantic_type)}
