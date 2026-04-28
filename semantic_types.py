import json
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from fastmcp import FastMCP, tool

from loader import load_data
from dtype import get_column_info


mcp = FastMCP("semantic-dtype-validator")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY
)


SYSTEM_PROMPT = """<KEEP SAME PROMPT>"""


# -----------------------------
# TOOL 1
# -----------------------------
@tool
def validate_semantic_types(dtype_info: dict):

    if not isinstance(dtype_info, dict) or not dtype_info:
        return {"report": [], "error": "Invalid dtype_info"}

    prompt = f"""
Column dtype information:

{json.dumps(dtype_info, indent=2)}
"""

    try:
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])

        content = response.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()

        llm_report = json.loads(content)

        if not isinstance(llm_report, list):
            raise ValueError("Not a list")

    except Exception as exc:
        return {
            "report": [],
            "error": f"LLM parsing failed: {str(exc)}"
        }

    llm_map = {item["column"]: item for item in llm_report if isinstance(item, dict)}

    final_report = []

    for column in dtype_info.keys():

        item = llm_map.get(column, {
            "column": column,
            "semantic_type": "unknown",
            "issue": "correct",
            "suggested_fix": "none",
            "confidence": 1.0,
            "reason": "no semantic dtype issue detected"
        })

        item.setdefault("semantic_type", "unknown")
        item.setdefault("issue", "correct")
        item.setdefault("suggested_fix", "none")
        item.setdefault("confidence", 0.9)
        item.setdefault("reason", "")

        final_report.append(item)

    return {"report": final_report}


# -----------------------------
# TOOL 2
# -----------------------------
@tool
def generate_semantic_report(file_path: str):

    if not file_path or not isinstance(file_path, str):
        return {"report": [], "error": "Invalid file_path"}

    try:
        result = load_data(file_path)
        if not result:
            return {"report": [], "error": "Failed to load data"}

        df = result[0]
    except Exception as e:
        return {"report": [], "error": str(e)}

    try:
        dtype_info = get_column_info(df)
    except Exception as e:
        return {"report": [], "error": f"dtype extraction failed: {str(e)}"}

    return validate_semantic_types(dtype_info)


# -----------------------------
# REGISTER TOOLS (explicit)
# -----------------------------
mcp.register_tool(validate_semantic_types)
mcp.register_tool(generate_semantic_report)


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    mcp.run()