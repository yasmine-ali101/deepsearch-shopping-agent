"""Pipeline orchestration, the agentic loop that ties the three agents together.

    query -> Agent 1 (rewrite) -> retrieve -> embed+FAISS -> Agent 2 (evaluate)
                  ^                                              |
                  |____________ refine, retry (max N) ___________|
                                                                 |
                                                     sufficient -> Agent 3 (answer)

The retry edge is what makes this agentic rather than a straight RAG chain: when
Agent 2 rejects a round, the query is refined and retrieval runs again instead of
handing weak evidence to the generator.

Collaborators (rewriter, answerer, retriever, index factory) are injected. Their
real implementations are imported lazily, so constructing a pipeline with stubs
costs no API key, no network, and no model download.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .agents.evaluator import evaluate
from .config import settings
from .models import Product

logger = logging.getLogger(__name__)


class Rewriter(Protocol):
    def rewrite(self, user_query: str) -> list[str]: ...


class AnswerWriter(Protocol):
    def answer(self, user_query: str, products: Sequence[Product]) -> str: ...


@dataclass
class PipelineResult:
    """Everything a caller needs, including how the pipeline got there."""

    answer: str
    products: list[Product] = field(default_factory=list)
    rounds_used: int = 0
    queries_used: list[str] = field(default_factory=list)
    succeeded: bool = False


class ShoppingPipeline:
    def __init__(
        self,
        rewriter: Rewriter | None = None,
        answerer: AnswerWriter | None = None,
        retriever: Callable[..., list[Product]] | None = None,
        index_factory: Callable[[list[Product]], Any] | None = None,
    ) -> None:
        if rewriter is None:
            from .agents.query_rewriter import QueryRewriter

            rewriter = QueryRewriter()
        if answerer is None:
            from .agents.answerer import Answerer

            answerer = Answerer()
        if retriever is None:
            from .retrieval.scraper import multi_search

            retriever = multi_search
        if index_factory is None:
            from .retrieval.index import ProductIndex

            index_factory = ProductIndex

        self._rewriter = rewriter
        self._answerer = answerer
        self._retriever = retriever
        self._index_factory = index_factory

    def run(
        self,
        user_query: str,
        max_rounds: int | None = None,
        threshold: float | None = None,
        country: str | None = None,
        k: int | None = None,
    ) -> PipelineResult:
        max_rounds = max_rounds or settings.max_rounds
        threshold = settings.relevance_threshold if threshold is None else threshold
        k = k or settings.top_k_per_query

        working_query = user_query
        all_queries: list[str] = []

        for round_index in range(1, max_rounds + 1):
            logger.info("Round %d/%d for query: %r", round_index, max_rounds, working_query)

            queries = self._rewriter.rewrite(working_query)
            all_queries.extend(queries)

            products = self._retriever(queries, k=k, country=country)
            if not products:
                logger.warning("Round %d retrieved no products", round_index)
                working_query = self._refine(working_query)
                continue

            hits = self._index_factory(products).search(queries, k=k)
            # `queries` is passed so each rewrite is judged against a threshold
            # matched to its script; see the evaluator module docstring.
            sufficient = evaluate(products, hits, threshold=threshold, queries=queries)

            if sufficient:
                logger.info("Round %d accepted %d products", round_index, len(sufficient))
                return PipelineResult(
                    answer=self._answerer.answer(user_query, sufficient),
                    products=sufficient,
                    rounds_used=round_index,
                    queries_used=all_queries,
                    succeeded=True,
                )

            logger.info("Round %d rejected by evaluator; refining query", round_index)
            working_query = self._refine(working_query)

        return PipelineResult(
            answer=(
                "I couldn't find results relevant enough to answer confidently. "
                "Try rephrasing with more detail (brand, budget, or use case), "
                "or lower the relevance threshold."
            ),
            rounds_used=max_rounds,
            queries_used=all_queries,
            succeeded=False,
        )

    @staticmethod
    def _refine(query: str) -> str:
        """Nudge the next rewrite toward more specific attributes."""
        suffix = "(refine: add product attributes, brand, or budget)"
        return query if suffix in query else f"{query} {suffix}"


def run_pipeline(user_query: str, **kwargs) -> str:
    """Convenience wrapper returning just the answer text."""
    return ShoppingPipeline().run(user_query, **kwargs).answer
