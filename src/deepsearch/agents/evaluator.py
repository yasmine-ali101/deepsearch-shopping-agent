"""Agent 2 — Relevance Evaluator.

The gatekeeper between retrieval and generation. It decides whether the products
that came back are good enough to answer with, or whether the pipeline should
refine the query and try again. This is what stops Agent 3 from confidently
summarising six irrelevant listings.

Depends only on numpy and `models.Product` — no LLM client, no vector library.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from ..config import settings
from ..models import Product

logger = logging.getLogger(__name__)


def filter_by_threshold(
    products: Sequence[Product],
    scores: np.ndarray,
    indices: np.ndarray,
    threshold: float | None = None,
) -> list[Product]:
    """Keep only the hits scoring at or above `threshold`, tagging each with its score."""
    threshold = settings.relevance_threshold if threshold is None else threshold

    sufficient: list[Product] = []
    for score, idx in zip(scores, indices):
        # FAISS pads with -1 when fewer than k neighbours exist; a negative index
        # would silently wrap around and return the wrong product.
        if idx < 0:
            continue
        product = products[idx]
        if score >= threshold:
            product.score = float(score)
            sufficient.append(product)
    return sufficient


def evaluate(
    products: Sequence[Product],
    search_results: Sequence[tuple[np.ndarray, np.ndarray]],
    threshold: float | None = None,
) -> list[Product]:
    """Collapse per-query hits into one deduplicated, relevance-ranked list.

    A product surfacing for several rewritten queries is a strong signal, but it
    should still appear once — ranked by the best score it earned, not by
    whichever query happened to run first.
    """
    threshold = settings.relevance_threshold if threshold is None else threshold

    best: dict[str, Product] = {}
    for scores, indices in search_results:
        for product in filter_by_threshold(products, scores, indices, threshold):
            existing = best.get(product.asin)
            if existing is None or (product.score or 0) > (existing.score or 0):
                best[product.asin] = product

    ranked = sorted(best.values(), key=lambda p: p.score or 0, reverse=True)
    logger.info(
        "Evaluator kept %d/%d products at threshold %.2f", len(ranked), len(products), threshold
    )
    return ranked
