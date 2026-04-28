class ValidationError(Exception):
    """Raised when a validation error occurs in the data debugger."""
    pass

class SemanticTypeError(ValidationError):
    """Raised when semantic dtype validation fails."""
    pass

class ToolError(Exception):
    """Generic tool-level error."""
    pass
