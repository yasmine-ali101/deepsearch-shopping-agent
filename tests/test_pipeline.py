"""Tests for the agentic retry loop.

Collaborators are injected as stubs, so this exercises the real orchestration
logic with no API key, no network, and no model download.
"""

import numpy as np
import pytest

from deepsearch.models import Product
from deepsearch.pipeline import ShoppingPipeline


class StubRewriter:
    def __init__(self):
        self.calls: list[str] = []

    def rewrite(self, user_query):
        self.calls.append(user_query)
        return [f"rewritten::{user_query}"]


class StubAnswerer:
    def answer(self, user_query, products):
        return f"answered {len(products)} product(s)"


def make_pipeline(products, scores):
    """Build a pipeline whose retrieval always returns `products` with `scores`."""

    class StubIndex:
        def __init__(self, _products):
            self._products = _products

        def search(self, queries, k=None):
            return [(np.array(scores), np.arange(len(scores)))]

    return ShoppingPipeline(
        rewriter=StubRewriter(),
        answerer=StubAnswerer(),
        retriever=lambda queries, k=None, country=None: list(products),
        index_factory=StubIndex,
    )


def test_succeeds_on_first_round_when_results_clear_threshold():
    products = [Product(keyword="q", asin="A1", title="Laptop")]
    pipeline = make_pipeline(products, [0.91])

    result = pipeline.run("cheap laptop", threshold=0.54)

    assert result.succeeded is True
    assert result.rounds_used == 1
    assert result.answer == "answered 1 product(s)"
    assert result.products[0].score == pytest.approx(0.91)


def test_retries_and_refines_the_query_when_the_evaluator_rejects():
    products = [Product(keyword="q", asin="A1", title="Unrelated")]
    pipeline = make_pipeline(products, [0.10])

    result = pipeline.run("cheap laptop", max_rounds=3, threshold=0.54)

    assert result.succeeded is False
    assert result.rounds_used == 3
    assert pipeline._rewriter.calls[0] == "cheap laptop"
    # The query is refined after the first rejection, then stays stable.
    assert "refine" in pipeline._rewriter.calls[1]
    assert pipeline._rewriter.calls[1] == pipeline._rewriter.calls[2]


def test_gives_up_gracefully_when_retrieval_returns_nothing():
    pipeline = make_pipeline([], [])

    result = pipeline.run("nonsense query", max_rounds=2)

    assert result.succeeded is False
    assert "couldn't find" in result.answer
    assert result.products == []


def test_records_every_query_it_tried():
    products = [Product(keyword="q", asin="A1")]
    pipeline = make_pipeline(products, [0.10])

    result = pipeline.run("headphones", max_rounds=2, threshold=0.9)

    assert len(result.queries_used) == 2
    assert all(q.startswith("rewritten::") for q in result.queries_used)
