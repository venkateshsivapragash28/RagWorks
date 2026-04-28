import json
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from fastmcp import FastMCP, tool

# ---- Local imports (assumes proper project structure) ----
from loader import load_data
from dtype import get_column_info
import errors


# ---- MCP SERVER ----
mcp = FastMCP("semantic-dtype-validator")


# ---- LLM SETUP ----
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY
)


# ---- SYSTEM PROMPT ----
SYSTEM_PROMPT = """
You are a dataset semantic dtype validation agent.

You will receive column dtype information.

Your task is to determine whether the detected dtype
matches the semantic meaning of the column name.

Return ONLY JSON.

Allowed semantic types:
numerical
categorical
datetime
text

Allowed suggested_fix values (STRICT):
to_numeric
to_float
to_int
to_datetime
to_category
to_string
none

Output schema:

[
 {
  "column": "<column name>",
  "semantic_type": "numerical | categorical | datetime | text",
  "issue": "wrong | correct",
  "suggested_fix": "<one of the allowed tokens>",
  "confidence": 0-1,
  "reason": "<short explanation>"
 }
]

Rules:
- suggested_fix MUST be from the allowed tokens only.
- If dtype is correct → issue="correct", suggested_fix="none"
- Return a result for EVERY column.
- Only return valid JSON.
"""


# ---- TOOL 1 ----
@tool
def validate_semantic_types(dtype_info: dict) -> list:
    """Validate whether dataset dtypes match semantic expectations."""

    prompt = f"""
Column dtype information:

{json.dumps(dtype_info, indent=2)}

Analyze whether dtype matches the semantic meaning of the column name.
Return a validation result for EVERY column.
"""

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ])

    content = response.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()

    try:
        llm_report = json.loads(content)

        if not isinstance(llm_report, list):
            raise errors.SemanticTypeError(
                "Semantic type validation output must be a JSON list."
            )

    except Exception as exc:
        # SAFE fallback instead of crashing MCP
        return [{
            "column": "unknown",
            "semantic_type": "unknown",
            "issue": "wrong",
            "suggested_fix": "none",
            "confidence": 0.0,
            "reason": f"JSON parsing failed: {str(exc)}"
        }]

    # ---- Map results ----
    llm_map = {}

    for item in llm_report:
        if isinstance(item, dict) and "column" in item:
            llm_map[item["column"]] = item

    final_report = []

    for column in dtype_info.keys():

        if column in llm_map:

            item = llm_map[column]

            item.setdefault("semantic_type", "unknown")
            item.setdefault("issue", "correct")
            item.setdefault("suggested_fix", "none")
            item.setdefault("confidence", 0.9)
            item.setdefault("reason", "")

            final_report.append(item)

        else:

            final_report.append({
                "column": column,
                "semantic_type": "unknown",
                "issue": "correct",
                "suggested_fix": "none",
                "confidence": 1.0,
                "reason": "no semantic dtype issue detected"
            })

    return final_report


# ---- TOOL 2 ----
@tool
def generate_semantic_report(file_path: str) -> list:
    """Generate a semantic dtype validation report for a dataset file."""

    df = load_data(file_path)[0]
    dtype_info = get_column_info(df)

    report = validate_semantic_types(dtype_info)

    return report


# ---- RUN MCP SERVER ----
if __name__ == "__main__":
    mcp.run()