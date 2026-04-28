import json
import sys
import os
import logging
from typing import Dict, List, Any, Optional

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
from fastmcp import tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY is not set")
    raise ValueError("GOOGLE_API_KEY environment variable is required")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0,
    google_api_key=GOOGLE_API_KEY,
    timeout=60
)


SEMANTIC_REASONER_PROMPT = """
You are a domain-agnostic semantic unit normalization expert.

Your task: Analyze messy numeric data and determine the standard unit and conversion factors.

You will receive:
- column_name: The name of the column
- unique_suffixes: List of detected unit suffixes
- unique_prefixes: List of detected prefixes (like currency symbols)
- pattern_types: Distribution of pattern types
- sample_values: A few example values

Your job:
1. Infer the semantic meaning of the column (age, mass, distance, storage, currency, etc.)
2. Determine the most appropriate standard unit
3. Provide conversion multipliers for each detected unit/suffix
4. Identify the fix strategy

CRITICAL RULES:
- Be domain-agnostic: Handle ANY type of measurement (physics, finance, biology, engineering, etc.)
- If you don't recognize a unit, make your best inference from context
- For scale suffixes (k, M, B, T), provide numeric multipliers
- For currency symbols, identify the currency type
- For percentages, convert to decimal (divide by 100)
- If units are incompatible, flag as "incompatible_units"

Output ONLY valid JSON in this exact format:

{
  "semantic_type": "<inferred type: age, mass, distance, storage, currency, percentage, duration, temperature, pressure, speed, energy, power, volume, area, count, ratio, score, rating, frequency, etc.>",
  "standard_unit": "<chosen standard unit>",
  "fix_strategy": "<unit_normalization | scale_normalization | currency_cleanup | percentage_conversion | incompatible_units | no_action>",
  "multipliers": {
    "suffix1": <multiplier as float>,
    "suffix2": <multiplier as float>
  },
  "confidence": <0.0 to 1.0>,
  "reasoning": "<brief explanation of your decision>",
  "warnings": ["<any warnings or concerns>"]
}

Examples:

Input: column="age", suffixes=["years", "yrs", "months", "days"]
Output:
{
  "semantic_type": "age",
  "standard_unit": "years",
  "fix_strategy": "unit_normalization",
  "multipliers": {
    "years": 1.0,
    "yrs": 1.0,
    "yr": 1.0,
    "months": 0.0833,
    "month": 0.0833,
    "days": 0.00274,
    "day": 0.00274
  },
  "confidence": 0.95,
  "reasoning": "Column name suggests age, units are time-based",
  "warnings": []
}

Input: column="storage_capacity", suffixes=["gb", "tb", "mb", "pb"]
Output:
{
  "semantic_type": "storage",
  "standard_unit": "GB",
  "fix_strategy": "unit_normalization",
  "multipliers": {
    "gb": 1.0,
    "tb": 1000.0,
    "mb": 0.001,
    "pb": 1000000.0
  },
  "confidence": 0.98,
  "reasoning": "Storage units detected, standardizing to GB",
  "warnings": []
}

Input: column="revenue", prefixes=["$"], suffixes=["k", "m", "b"]
Output:
{
  "semantic_type": "currency",
  "standard_unit": "USD",
  "fix_strategy": "scale_normalization",
  "multipliers": {
    "k": 1000.0,
    "m": 1000000.0,
    "b": 1000000000.0
  },
  "confidence": 0.90,
  "reasoning": "Currency with scale suffixes, standardizing to base USD",
  "warnings": []
}

Input: column="efficiency", suffixes=["%"]
Output:
{
  "semantic_type": "percentage",
  "standard_unit": "decimal",
  "fix_strategy": "percentage_conversion",
  "multipliers": {
    "%": 0.01
  },
  "confidence": 1.0,
  "reasoning": "Percentage values, converting to decimal",
  "warnings": []
}

Input: column="mixed_data", suffixes=["kg", "years", "meters"]
Output:
{
  "semantic_type": "unknown",
  "standard_unit": "none",
  "fix_strategy": "incompatible_units",
  "multipliers": {},
  "confidence": 0.0,
  "reasoning": "Incompatible units detected (mass, time, distance)",
  "warnings": ["Column contains incompatible unit types", "Manual review required"]
}

Be creative and handle ANY domain: medical, financial, scientific, engineering, sports, etc.
"""


@tool(
    name="infer_semantic_type",
    description="Infer semantic type and normalization rules for messy numeric column values.",
    input_schema={"column_name": "string", "unique_suffixes": "array", "unique_prefixes": "array", "pattern_types": "object", "sample_values": "array"},
    output_schema={"semantic_info": "object"},
)
def infer_semantic_type(column_name: str, 
                       unique_suffixes: List[str],
                       unique_prefixes: List[str],
                       pattern_types: Dict[str, int],
                       sample_values: List[str]) -> Optional[Dict[str, Any]]:
    """
    Use LLM to infer semantic type and conversion rules.
    
    Args:
        column_name: Name of the column
        unique_suffixes: List of unique unit suffixes
        unique_prefixes: List of unique prefixes
        pattern_types: Distribution of pattern types
        sample_values: Sample values from the column
        
    Returns:
        Dictionary containing semantic type, standard unit, and multipliers
    """
    if not unique_suffixes and not unique_prefixes:
        logger.warning(f"No suffixes or prefixes found for column '{column_name}'")
        return None
    
    prompt = f"""
Analyze this column:

Column Name: {column_name}

Detected Suffixes: {json.dumps(unique_suffixes)}
Detected Prefixes: {json.dumps(unique_prefixes)}
Pattern Types: {json.dumps(pattern_types)}
Sample Values: {json.dumps(sample_values[:10])}

Determine the semantic type, standard unit, and conversion multipliers.
Return ONLY valid JSON.
"""
    
    try:
        response = llm.invoke([
            SystemMessage(content=SEMANTIC_REASONER_PROMPT),
            HumanMessage(content=prompt)
        ])
    except Exception as e:
        logger.error(f"LLM API call failed for column '{column_name}': {str(e)}")
        raise RuntimeError(f"Failed to invoke LLM: {str(e)}") from e
    
    if not response or not hasattr(response, 'content'):
        logger.error("Invalid LLM response structure")
        raise ValueError("LLM returned invalid response")
    
    content = response.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()
    
    try:
        semantic_info = json.loads(content)
        
        if not isinstance(semantic_info, dict):
            logger.error(f"LLM returned non-dict response: {type(semantic_info)}")
            return None
        
        required_fields = ["semantic_type", "standard_unit", "fix_strategy", "multipliers"]
        for field in required_fields:
            if field not in semantic_info:
                logger.warning(f"Missing field '{field}' in LLM response, adding default")
                semantic_info.setdefault(field, None if field != "multipliers" else {})
        
        semantic_info.setdefault("confidence", 0.5)
        semantic_info.setdefault("reasoning", "")
        semantic_info.setdefault("warnings", [])
        
        logger.info(f"Inferred semantic type for '{column_name}': {semantic_info['semantic_type']}")
        return semantic_info
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM JSON response: {str(e)}")
        logger.debug(f"Raw content: {content[:500]}")
        return None


@tool(
    name="batch_infer_semantic_types",
    description="Infer semantic types for multiple columns using detected patterns.",
    input_schema={"column_patterns": "object"},
    output_schema={"semantic_info": "object"},
)
def batch_infer_semantic_types(column_patterns: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Infer semantic types for multiple columns.
    
    Args:
        column_patterns: Dictionary mapping column names to their pattern discovery results
        
    Returns:
        Dictionary mapping column names to their semantic inference results
    """
    if not column_patterns:
        logger.warning("Empty column_patterns provided")
        return {}
    
    results = {}
    
    for col_name, patterns in column_patterns.items():
        try:
            unique_suffixes = patterns.get("unique_suffixes", [])
            unique_prefixes = patterns.get("unique_prefixes", [])
            pattern_types = patterns.get("pattern_types", {})
            
            sample_values = []
            if "samples" in patterns:
                sample_values = patterns["samples"]
            
            semantic_info = infer_semantic_type(
                col_name,
                unique_suffixes,
                unique_prefixes,
                pattern_types,
                sample_values
            )
            
            if semantic_info:
                results[col_name] = semantic_info
            else:
                logger.warning(f"Failed to infer semantic type for column '{col_name}'")
                
        except Exception as e:
            logger.error(f"Error processing column '{col_name}': {str(e)}")
            continue
    
    logger.info(f"Successfully inferred semantic types for {len(results)} columns")
    return results
