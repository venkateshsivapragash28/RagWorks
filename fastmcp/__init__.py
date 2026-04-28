from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

registry: List["ToolSpec"] = []

@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    fn: Callable[..., Any]


def tool(
    name: Optional[str] = None,
    description: str = "",
    input_schema: Optional[Dict[str, Any]] = None,
    output_schema: Optional[Dict[str, Any]] = None,
):
    """Decorator to register a function as a FastMCP tool."""
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        spec = ToolSpec(
            name=name or fn.__name__,
            description=description or (fn.__doc__.strip().splitlines()[0] if fn.__doc__ else ""),
            input_schema=input_schema or {},
            output_schema=output_schema or {},
            fn=fn,
        )
        registry.append(spec)

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        wrapper._fastmcp_tool = spec  # type: ignore[attr-defined]
        return wrapper
    return decorator


def get_registered_tools() -> List[ToolSpec]:
    return list(registry)
