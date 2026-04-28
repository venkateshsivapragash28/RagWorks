# Data Debugger FastMCP Tools

This workspace now converts every Python source file into a FastMCP-style tool endpoint.

## Files converted to FastMCP tools

- `dtype.py`
- `label_fixer.py`
- `label_validator.py`
- `loader.py`
- `metadata_extractor.py`
- `pattern_extractor.py`
- `semantic_reasoner.py`
- `semantic_types.py`
- `splitter.py`
- `statistical_guardrails.py`
- `structure_analyzer.py`
- `vectorized_cleaner.py`
- `data_debugger_fastmcp.py`

## Tool manifest

- `fastmcp_manifest.json` contains the full tool registry with every tool name.
- `fastmcp/__init__.py` provides the FastMCP decorator and registry stub.

## Example usage

```python
from data_debugger_fastmcp import load_dataset, get_column_schema

result = load_dataset(r"C:\path\to\dataset.csv")
print(result["metadata"])
print(result["preview"])
```

## Notes

- Every module now exposes at least one `@tool(...)` FastMCP endpoint.
- Some tools depend on packages like `pandas`, `numpy`, and LLM support libraries.
- If dependencies are missing, install them before using the tools.
