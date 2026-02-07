"""Custom exceptions for the AnyBURL rule learning system."""


class AnyBURLError(Exception):
    """Base exception for all AnyBURL errors."""


class GraphStructureError(AnyBURLError):
    """Raised when the knowledge graph has an invalid or empty structure.

    Examples include no edge types defined, empty edge indices,
    or mismatched dimensions between node types and edge indices.
    """


class RuleGeneralizationError(AnyBURLError):
    """Raised when a concrete path cannot be generalized into a valid rule.

    This indicates a structural problem with the path, such as
    insufficient length or inconsistent node types.
    """


class InsufficientDataError(AnyBURLError):
    """Raised when there is not enough data to compute a metric.

    For example, head coverage requires at least one triple with the
    given relation in the knowledge graph.
    """
