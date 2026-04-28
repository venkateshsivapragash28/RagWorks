import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List
import re
from fastmcp import tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorizedCleaner:
    """Applies semantic normalization rules using vectorized pandas operations."""
    
    def __init__(self):
        self.pattern_extractor = None
    
    def _extract_number_vectorized(self, series: pd.Series) -> pd.Series:
        """Extract numeric part from strings vectorized."""
        def extract_num(val):
            if pd.isna(val):
                return np.nan
            
            val_str = str(val).strip()
            
            # Remove currency symbols
            val_str = re.sub(r'[\$\€\£\₹]', '', val_str)
            
            # Remove commas
            val_str = val_str.replace(',', '')
            
            # Extract first number (including decimals)
            match = re.search(r'[+-]?\d+\.?\d*', val_str)
            if match:
                try:
                    return float(match.group())
                except:
                    return np.nan
            return np.nan
        
        return series.apply(extract_num)
    
    def _extract_suffix_vectorized(self, series: pd.Series) -> pd.Series:
        """Extract suffix/unit from strings vectorized."""
        def extract_suf(val):
            if pd.isna(val):
                return None
            
            val_str = str(val).strip()
            
            # Remove currency symbols and numbers
            val_str = re.sub(r'[\$\€\£\₹]', '', val_str)
            val_str = re.sub(r'[+-]?\d+\.?\d*', '', val_str)
            
            # Clean and lowercase
            suffix = val_str.strip().lower()
            
            return suffix if suffix else None
        
        return series.apply(extract_suf)
    
    def _extract_prefix_vectorized(self, series: pd.Series) -> pd.Series:
        """Extract prefix (like currency symbol) vectorized."""
        def extract_pre(val):
            if pd.isna(val):
                return None
            
            val_str = str(val).strip()
            
            # Check for currency symbols
            match = re.match(r'^([\$\€\£\₹])', val_str)
            if match:
                return match.group(1)
            return None
        
        return series.apply(extract_pre)
    
    def apply_unit_normalization(self, 
                                 series: pd.Series, 
                                 multipliers: Dict[str, float],
                                 standard_unit: str) -> pd.Series:
        """
        Apply unit normalization using multipliers.
        
        Args:
            series: Original series with messy units
            multipliers: Dictionary mapping suffixes to multipliers
            standard_unit: Target standard unit
            
        Returns:
            Cleaned series with normalized values
        """
        if series.empty:
            logger.warning("Empty series provided")
            return series
        
        if not multipliers:
            logger.warning("No multipliers provided")
            return series
        
        try:
            # Extract numbers and suffixes
            numbers = self._extract_number_vectorized(series)
            suffixes = self._extract_suffix_vectorized(series)
            
            # Map suffixes to multipliers
            multiplier_series = suffixes.map(multipliers)
            
            # Fill unmapped suffixes with 1.0 (assume already in standard unit)
            multiplier_series = multiplier_series.fillna(1.0)
            
            # Apply conversion
            cleaned = numbers * multiplier_series
            
            logger.info(f"Applied unit normalization: {len(cleaned.dropna())} values converted to {standard_unit}")
            return cleaned
            
        except Exception as e:
            logger.error(f"Failed to apply unit normalization: {str(e)}")
            raise RuntimeError(f"Unit normalization failed: {str(e)}") from e
    
    def apply_scale_normalization(self,
                                  series: pd.Series,
                                  multipliers: Dict[str, float],
                                  standard_unit: str) -> pd.Series:
        """
        Apply scale normalization (k, M, B, T).
        
        Args:
            series: Original series with scale suffixes
            multipliers: Dictionary mapping scale suffixes to multipliers
            standard_unit: Target unit
            
        Returns:
            Cleaned series with expanded values
        """
        if series.empty:
            logger.warning("Empty series provided")
            return series
        
        if not multipliers:
            logger.warning("No multipliers provided")
            return series
        
        try:
            numbers = self._extract_number_vectorized(series)
            suffixes = self._extract_suffix_vectorized(series)
            
            # Map scale suffixes
            multiplier_series = suffixes.map(multipliers)
            multiplier_series = multiplier_series.fillna(1.0)
            
            cleaned = numbers * multiplier_series
            
            logger.info(f"Applied scale normalization: {len(cleaned.dropna())} values normalized")
            return cleaned
            
        except Exception as e:
            logger.error(f"Failed to apply scale normalization: {str(e)}")
            raise RuntimeError(f"Scale normalization failed: {str(e)}") from e
    
    def apply_percentage_conversion(self, series: pd.Series) -> pd.Series:
        """
        Convert percentages to decimals.
        
        Args:
            series: Series with percentage values
            
        Returns:
            Series with decimal values
        """
        if series.empty:
            logger.warning("Empty series provided")
            return series
        
        try:
            numbers = self._extract_number_vectorized(series)
            cleaned = numbers / 100.0
            
            logger.info(f"Applied percentage conversion: {len(cleaned.dropna())} values converted")
            return cleaned
            
        except Exception as e:
            logger.error(f"Failed to apply percentage conversion: {str(e)}")
            raise RuntimeError(f"Percentage conversion failed: {str(e)}") from e
    
    def apply_currency_cleanup(self,
                               series: pd.Series,
                               multipliers: Dict[str, float],
                               standard_unit: str) -> pd.Series:
        """
        Clean currency values and apply scale multipliers.
        
        Args:
            series: Series with currency values
            multipliers: Scale multipliers (k, M, B)
            standard_unit: Currency code
            
        Returns:
            Cleaned numeric series
        """
        if series.empty:
            logger.warning("Empty series provided")
            return series
        
        try:
            numbers = self._extract_number_vectorized(series)
            suffixes = self._extract_suffix_vectorized(series)
            
            if multipliers:
                multiplier_series = suffixes.map(multipliers)
                multiplier_series = multiplier_series.fillna(1.0)
                cleaned = numbers * multiplier_series
            else:
                cleaned = numbers
            
            logger.info(f"Applied currency cleanup: {len(cleaned.dropna())} values cleaned")
            return cleaned
            
        except Exception as e:
            logger.error(f"Failed to apply currency cleanup: {str(e)}")
            raise RuntimeError(f"Currency cleanup failed: {str(e)}") from e
    
    def clean_column(self,
                    series: pd.Series,
                    semantic_info: Dict[str, Any]) -> pd.Series:
        """
        Clean a column based on semantic inference results.
        
        Args:
            series: Original series to clean
            semantic_info: Semantic inference results from LLM
            
        Returns:
            Cleaned series
        """
        if not isinstance(series, pd.Series):
            raise TypeError(f"Expected Series, got {type(series)}")
        
        if not semantic_info:
            logger.warning("No semantic info provided, returning original series")
            return series
        
        fix_strategy = semantic_info.get("fix_strategy", "no_action")
        multipliers = semantic_info.get("multipliers", {})
        standard_unit = semantic_info.get("standard_unit", "unknown")
        
        logger.info(f"Applying fix strategy: {fix_strategy}")
        
        try:
            if fix_strategy == "unit_normalization":
                return self.apply_unit_normalization(series, multipliers, standard_unit)
            
            elif fix_strategy == "scale_normalization":
                return self.apply_scale_normalization(series, multipliers, standard_unit)
            
            elif fix_strategy == "percentage_conversion":
                return self.apply_percentage_conversion(series)
            
            elif fix_strategy == "currency_cleanup":
                return self.apply_currency_cleanup(series, multipliers, standard_unit)
            
            elif fix_strategy == "incompatible_units":
                logger.warning("Incompatible units detected, returning original series")
                return series
            
            elif fix_strategy == "no_action":
                logger.info("No action required")
                return series
            
            else:
                logger.warning(f"Unknown fix strategy: {fix_strategy}")
                return series
                
        except Exception as e:
            logger.error(f"Failed to clean column: {str(e)}")
            raise RuntimeError(f"Column cleaning failed: {str(e)}") from e
    
    def clean_dataframe(self,
                       df: pd.DataFrame,
                       semantic_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Clean multiple columns in a DataFrame.
        
        Args:
            df: Original DataFrame
            semantic_results: Dictionary mapping column names to semantic info
            
        Returns:
            DataFrame with cleaned columns
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(df)}")
        
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return df
        
        if not semantic_results:
            logger.warning("No semantic results provided")
            return df
        
        df_cleaned = df.copy()
        
        for col_name, semantic_info in semantic_results.items():
            if col_name not in df_cleaned.columns:
                logger.warning(f"Column '{col_name}' not found in DataFrame")
                continue
            
            try:
                cleaned_series = self.clean_column(df_cleaned[col_name], semantic_info)
                df_cleaned[col_name] = cleaned_series
                logger.info(f"Successfully cleaned column '{col_name}'")
                
            except Exception as e:
                logger.error(f"Failed to clean column '{col_name}': {str(e)}")
                continue
        
        logger.info(f"Cleaned {len(semantic_results)} columns")
        return df_cleaned


@tool(
    name="apply_unit_normalization",
    description="Normalize a list of values with unit suffix multipliers.",
    input_schema={"values": "array", "multipliers": "object", "standard_unit": "string"},
    output_schema={"cleaned_values": "array"},
)
def apply_unit_normalization_tool(values: List[Any], multipliers: Dict[str, float], standard_unit: str):
    cleaner = VectorizedCleaner()
    series = pd.Series(values)
    cleaned = cleaner.apply_unit_normalization(series, multipliers, standard_unit)
    return {"cleaned_values": cleaned.tolist()}


@tool(
    name="apply_percentage_conversion",
    description="Convert percentage-like values to decimals.",
    input_schema={"values": "array"},
    output_schema={"cleaned_values": "array"},
)
def apply_percentage_conversion_tool(values: List[Any]):
    cleaner = VectorizedCleaner()
    series = pd.Series(values)
    cleaned = cleaner.apply_percentage_conversion(series)
    return {"cleaned_values": cleaned.tolist()}


@tool(
    name="clean_column",
    description="Clean a list of values using semantic fix strategy information.",
    input_schema={"values": "array", "semantic_info": "object"},
    output_schema={"cleaned_values": "array"},
)
def clean_column_tool(values: List[Any], semantic_info: Dict[str, Any]):
    cleaner = VectorizedCleaner()
    series = pd.Series(values)
    cleaned = cleaner.clean_column(series, semantic_info)
    return {"cleaned_values": cleaned.tolist()}
