"""Exception hierarchy for AnyBURL.

Each error also subclasses the built-in exception it replaces, so callers
catching ``ValueError`` or ``RuntimeError`` keep working.
"""

__all__ = [
    "AnyBURLError",
    "ConfigurationError",
    "GraphSchemaError",
    "InvalidRuleError",
    "NotFittedError",
]


class AnyBURLError(Exception):
    """Base class for all errors raised by AnyBURL."""


class ConfigurationError(AnyBURLError, ValueError):
    """A configuration value or argument is outside its valid range."""


class GraphSchemaError(AnyBURLError, ValueError):
    """A node type or edge type is missing from, or empty in, the graph."""


class InvalidRuleError(AnyBURLError, ValueError):
    """A term, path or body chain does not form a well-formed rule."""


class NotFittedError(AnyBURLError, RuntimeError):
    """A pipeline method was called before ``fit`` produced any rules."""
