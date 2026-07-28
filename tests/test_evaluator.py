"""Tests for Agent 2 — the relevance gate.

These run offline: no Cohere key, no network, no model download.
"""

import numpy as np
import pytest

from deepsearch.agents.evaluator import evaluate, filter_by_threshold
from deepsearch.models import Product


def make_products(n: int) -> list[Product]:
    return [Product(keyword="q", asin=f"ASIN{i}", title=f"Product {i}") for i in range(n)]


def test_filter_keeps_only_scores_at_or_above_threshold():
    products = make_products(3)
    scores = np.array([0.9, 0.54, 0.2])
    indices = np.array([0, 1, 2])

    kept = filter_by_threshold(products, scores, indices, threshold=0.54)

    assert [p.asin for p in kept] == ["ASIN0", "ASIN1"]
    assert kept[0].score == pytest.approx(0.9)


def test_filter_ignores_faiss_padding_indices():
    """FAISS returns -1 when fewer than k neighbours exist; that must not wrap around."""
    products = make_products(2)
    scores = np.array([0.8, 0.7])
    indices = np.array([0, -1])

    kept = filter_by_threshold(products, scores, indices, threshold=0.5)

    assert [p.asin for p in kept] == ["ASIN0"]


def test_evaluate_deduplicates_and_keeps_the_highest_score():
    products = make_products(2)
    # ASIN0 surfaces for both queries, scoring 0.6 then 0.85.
    results = [
        (np.array([0.60, 0.55]), np.array([0, 1])),
        (np.array([0.85]), np.array([0])),
    ]

    ranked = evaluate(products, results, threshold=0.5)

    assert [p.asin for p in ranked] == ["ASIN0", "ASIN1"]
    assert ranked[0].score == pytest.approx(0.85)


def test_evaluate_returns_empty_when_nothing_clears_the_bar():
    products = make_products(2)
    results = [(np.array([0.10, 0.20]), np.array([0, 1]))]

    assert evaluate(products, results, threshold=0.54) == []
