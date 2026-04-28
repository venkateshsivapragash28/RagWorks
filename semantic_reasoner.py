import json
import os
import logging
from typing import Dict, List, Any

from fastmcp import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---- LLM SETUP (SAFE) ----
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0,
    google_api_key=GOOGLE_API_KEY,
    timeout=60
)


# ---- PROMPT (UNCHANGED) ----
SEMANTIC_REASONER_PROMPT = """<KEEP YOUR FULL PROMPT AS IS>"""


# -----------------------------
# CORE FUNCTION (SAFE WRAPPED)
# -----------------------------
def _infer_semantic_type_core(
    column_name: str,
    unique_suffixes: List[str],
    unique_prefixes: List[str],
    pattern_types: Dict[str, int],
    sample_values: List[str]
) -> Dict[str, Any]:

    if not unique_suffixes and not unique_prefixes:
        return {
            "semantic_type": "unknown",
            "standard_unit": "none",
            "fix_strategy": "no_action",
            "multipliers": {},
            "confidence": 0.0,
            "reasoning": "No detectable patterns",
            "warnings": ["No suffixes or prefixes found"]
        }

    prompt = f"""
Column Name: {column_name}

Suffixes: {json.dumps(unique_suffixes)}
Prefixes: {json.dumps(unique_prefixes)}
Pattern Types: {json.dumps(pattern_types)}
Sample Values: {json.dumps(sample_values[:10])}
"""

    try:
        response = llm.invoke([
            SystemMessage(content=SEMANTIC_REASONER_PROMPT),
            HumanMessage(content=prompt)
        ])
    except Exception as e:
        return {
            "error": f"LLM failed: {str(e)}"
        }

    if not response or not hasattr(response, "content"):
        return {"error": "Invalid LLM response"}

    content = response.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()

    try:
        semantic_info = json.loads(content)
    except Exception:
        return {
            "error": "Invalid JSON from LLM",
            "raw_output": content[:300]
        }

    # ---- enforce schema ----
    semantic_info.setdefault("semantic_type", "unknown")
    semantic_info.setdefault("standard_unit", "none")
    semantic_info.setdefault("fix_strategy", "no_action")
    semantic_info.setdefault("multipliers", {})
    semantic_info.setdefault("confidence", 0.5)
    semantic_info.setdefault("reasoning", "")
    semantic_info.setdefault("warnings", [])

    return semantic_info


# -----------------------------
# MCP TOOL 1
# -----------------------------
@tool(
    name="infer_semantic_type",
    description="Infer semantic type and normalization rules for a column.",
    input_schema={
        "column_name": "string",
        "unique_suffixes": "array",
        "unique_prefixes": "array",
        "pattern_types": "object",
        "sample_values": "array"
    },
    output_schema={"semantic_info": "object"},
)
def infer_semantic_type(
    column_name: str,
    unique_suffixes: List[str],
    unique_prefixes: List[str],
    pattern_types: Dict[str, int],
    sample_values: List[str]
):

    result = _infer_semantic_type_core(
        column_name,
        unique_suffixes,
        unique_prefixes,
        pattern_types,
        sample_values
    )

    return {"semantic_info": result}


# -----------------------------
# MCP TOOL 2 (BATCH)
# -----------------------------
@tool(
    name="batch_infer_semantic_types",
    description="Infer semantic types for multiple columns.",
    input_schema={"column_patterns": "object"},
    output_schema={"semantic_info": "object"},
)
def batch_infer_semantic_types(
    column_patterns: Dict[str, Dict[str, Any]]
):

    if not isinstance(column_patterns, dict):
        return {"semantic_info": {}}

    results = {}

    for col_name, patterns in column_patterns.items():

        try:
            result = _infer_semantic_type_core(
                col_name,
                patterns.get("unique_suffixes", []),
                patterns.get("unique_prefixes", []),
                patterns.get("pattern_types", {}),
                patterns.get("samples", [])
            )

            results[col_name] = result

        except Exception as e:
            results[col_name] = {
                "error": str(e)
            }

    return {"semantic_info": results}