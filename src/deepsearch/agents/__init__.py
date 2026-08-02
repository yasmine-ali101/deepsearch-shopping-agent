"""Agent implementations.

`evaluate` is imported eagerly (numpy only). The LLM-backed agents are exposed
lazily so that importing this package doesn't require the `cohere` client, the
evaluator and its tests have no business paying for that.
"""

from .evaluator import evaluate, filter_by_threshold, is_arabic, threshold_for

__all__ = [
    "evaluate",
    "filter_by_threshold",
    "is_arabic",
    "threshold_for",
    "QueryRewriter",
    "Answerer",
]


def __getattr__(name: str):
    if name == "QueryRewriter":
        from .query_rewriter import QueryRewriter

        return QueryRewriter
    if name == "Answerer":
        from .answerer import Answerer

        return Answerer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
