import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
from fastmcp import tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatternExtractor:
    """Extracts patterns from messy numeric columns containing units, scales, and formats."""
    
    PATTERNS = {
        "number_suffix": r'^([+-]?\d+\.?\d*)\s*([a-zA-Z%]+)$',
        "prefix_number": r'^([\$\€\£\₹])\s*([+-]?\d+\.?\d*)$',
        "number_scale": r'^([+-]?\d+\.?\d*)\s*([kKmMbBtT])$',
        "percentage": r'^([+-]?\d+\.?\d*)\s*%$',
        "range": r'^(\d+\.?\d*)\s*[-to]+\s*(\d+\.?\d*)$',
        "fraction": r'^(\d+)/(\d+)$',
        "complex_unit": r'^([+-]?\d+\.?\d*)\s*([a-zA-Z]+/[a-zA-Z]+)$',
        "scientific": r'^([+-]?\d+\.?\d*)[eE]([+-]?\d+)$',
        "currency_formatted": r'^([\$\€\£\₹])\s*([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)$',
        "plain_number": r'^([+-]?\d+\.?\d*)$'
    }
    
    def __init__(self):
        self.compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.PATTERNS.items()
        }
    
    def extract_pattern(self, value: str) -> Optional[Dict[str, Any]]:
        """Extract pattern from a single value."""
        if pd.isna(value) or value is None:
            return None
        
        value_str = str(value).strip()
        
        if not value_str:
            return None
        
        for pattern_name, pattern_regex in self.compiled_patterns.items():
            match = pattern_regex.match(value_str)
            
            if match:
                groups = match.groups()
                
                if pattern_name == "number_suffix":
                    return {
                        "pattern_type": "number_suffix",
                        "number": groups[0],
                        "suffix": groups[1].lower(),
                        "prefix": None,
                        "original": value_str
                    }
                
                elif pattern_name == "prefix_number":
                    return {
                        "pattern_type": "prefix_number",
                        "number": groups[1],
                        "suffix": None,
                        "prefix": groups[0],
                        "original": value_str
                    }
                
                elif pattern_name == "number_scale":
                    return {
                        "pattern_type": "number_scale",
                        "number": groups[0],
                        "suffix": groups[1].lower(),
                        "prefix": None,
                        "scale": groups[1].lower(),
                        "original": value_str
                    }
                
                elif pattern_name == "percentage":
                    return {
                        "pattern_type": "percentage",
                        "number": groups[0],
                        "suffix": "%",
                        "prefix": None,
                        "original": value_str
                    }
                
                elif pattern_name == "range":
                    avg = (float(groups[0]) + float(groups[1])) / 2
                    return {
                        "pattern_type": "range",
                        "number": str(avg),
                        "suffix": None,
                        "prefix": None,
                        "original": value_str
                    }
                
                elif pattern_name == "fraction":
                    result = float(groups[0]) / float(groups[1])
                    return {
                        "pattern_type": "fraction",
                        "number": str(result),
                        "suffix": None,
                        "prefix": None,
                        "original": value_str
                    }
                
                elif pattern_name == "complex_unit":
                    return {
                        "pattern_type": "complex_unit",
                        "number": groups[0],
                        "suffix": groups[1].lower(),
                        "prefix": None,
                        "original": value_str
                    }
                
                elif pattern_name == "scientific":
                    return {
                        "pattern_type": "scientific",
                        "number": str(float(groups[0]) * (10 ** float(groups[1]))),
                        "suffix": None,
                        "prefix": None,
                        "original": value_str
                    }
                
                elif pattern_name == "currency_formatted":
                    clean_number = groups[1].replace(',', '')
                    return {
                        "pattern_type": "currency_formatted",
                        "number": clean_number,
                        "suffix": None,
                        "prefix": groups[0],
                        "original": value_str
                    }
                
                elif pattern_name == "plain_number":
                    return {
                        "pattern_type": "plain_number",
                        "number": groups[0],
                        "suffix": None,
                        "prefix": None,
                        "original": value_str
                    }
        
        return {
            "pattern_type": "unparseable",
            "number": None,
            "suffix": None,
            "prefix": None,
            "original": value_str
        }
    
    def extract_column_patterns(self, series: pd.Series) -> pd.DataFrame:
        """Extract patterns from all values in a column."""
        if not isinstance(series, pd.Series):
            raise TypeError(f"Expected Series, got {type(series)}")
        
        patterns = []
        
        for value in series.dropna():
            pattern = self.extract_pattern(value)
            if pattern:
                patterns.append(pattern)
        
        if not patterns:
            logger.warning(f"No patterns extracted from series")
            return pd.DataFrame()
        
        df_patterns = pd.DataFrame(patterns)
        logger.info(f"Extracted {len(df_patterns)} patterns from {len(series)} values")
        
        return df_patterns
    
    def discover_unique_patterns(self, df_patterns: pd.DataFrame) -> Dict[str, Any]:
        """Discover unique suffixes, prefixes, and pattern types."""
        if df_patterns.empty:
            return {
                "unique_suffixes": [],
                "unique_prefixes": [],
                "pattern_types": {},
                "suffix_counts": {},
                "prefix_counts": {}
            }
        
        suffixes = df_patterns['suffix'].dropna()
        unique_suffixes = suffixes.unique().tolist()
        suffix_counts = Counter(suffixes).most_common()
        
        prefixes = df_patterns['prefix'].dropna()
        unique_prefixes = prefixes.unique().tolist()
        prefix_counts = Counter(prefixes).most_common()
        
        pattern_types = Counter(df_patterns['pattern_type']).most_common()
        
        discovery = {
            "unique_suffixes": unique_suffixes,
            "unique_prefixes": unique_prefixes,
            "pattern_types": dict(pattern_types),
            "suffix_counts": dict(suffix_counts),
            "prefix_counts": dict(prefix_counts),
            "total_patterns": len(df_patterns),
            "unparseable_count": len(df_patterns[df_patterns['pattern_type'] == 'unparseable'])
        }
        
        logger.info(f"Discovered {len(unique_suffixes)} unique suffixes, "
                   f"{len(unique_prefixes)} unique prefixes")
        
        return discovery


@tool(
    name="extract_pattern",
    description="Extract a numeric pattern from a single value.",
    input_schema={"value": "string"},
    output_schema={"pattern": "object"},
)
def extract_pattern_tool(value: str):
    return {"pattern": PatternExtractor().extract_pattern(value)}


@tool(
    name="extract_column_patterns",
    description="Extract patterns from a list of values representing a dataset column.",
    input_schema={"values": "array"},
    output_schema={"patterns": "array"},
)
def extract_column_patterns_tool(values: List[str]):
    extractor = PatternExtractor()
    series = pd.Series(values)
    patterns = extractor.extract_column_patterns(series)
    return {"patterns": patterns.to_dict(orient="records") if not patterns.empty else []}


@tool(
    name="discover_unique_patterns",
    description="Discover unique suffixes, prefixes, and pattern types from extracted patterns.",
    input_schema={"patterns": "array"},
    output_schema={"summary": "object"},
)
def discover_unique_patterns_tool(patterns: List[Dict[str, Any]]):
    df_patterns = pd.DataFrame(patterns)
    summary = PatternExtractor().discover_unique_patterns(df_patterns)
    return {"summary": summary}
