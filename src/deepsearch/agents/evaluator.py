"""Agent 2, the Relevance Evaluator.

The gatekeeper between retrieval and generation. It decides whether the products
that came back are good enough to answer with, or whether the pipeline should
refine the query and try again. This is what stops Agent 3 from confidently
summarising six irrelevant listings.

Depends only on numpy and `models.Product`, so no LLM client and no vector
library are needed to test it.

## Why the threshold is language aware

Agent 1 emits three English and three Arabic rewrites, and every one is scored
against the same index. Measured on ten matched query pairs meaning the same
thing, against the same English product text:

    mean cosine similarity to the correct product
        English  0.678
        Arabic   0.571      gap 0.107

At a single fixed 0.54 threshold, English rewrites passed 10/10 and Arabic only
7/10. The gate was therefore throwing away the Arabic half of the retrieval that
the bilingual design exists to provide. Scoring Arabic queries against a
correspondingly lower bar keeps the comparison fair.

The offset is a property of the encoder and the catalogue language rather than a
universal constant. Re-measure it with `scripts/calibrate_threshold.py`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from ..config import settings
from ..models import Product

logger = logging.getLogger(__name__)

# Arabic, Arabic Supplement, and Arabic Extended-A blocks.
_ARABIC_RANGES = ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF))


def is_arabic(text: str) -> bool:
    """True when the text is predominantly Arabic script.

    A majority test over letters rather than "contains any Arabic character", so
    a mostly English query naming one Arabic brand is still treated as English.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    arabic = sum(
        1 for c in letters if any(lo <= ord(c) <= hi for lo, hi in _ARABIC_RANGES)
    )
    return arabic / len(letters) > 0.5


def threshold_for(query: str, base_threshold: float | None = None) -> float:
    """The relevance bar this query should be held to."""
    base = settings.relevance_threshold if base_threshold is None else base_threshold
    return base - settings.arabic_threshold_offset if is_arabic(query) else base


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
        # FAISS pads with -1 when fewer than k neighbours exist, and a negative
        # index would silently wrap around and return the wrong product.
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
    queries: Sequence[str] | None = None,
) -> list[Product]:
    """Collapse per-query hits into one deduplicated, relevance-ranked list.

    When `queries` is supplied, each result set is judged against a threshold
    appropriate to that query's script. Without it every set is judged against
    the base threshold, which is the older behaviour and biased against Arabic.

    A product surfacing for several rewritten queries is a strong signal, but it
    should still appear once, ranked by the best score it earned rather than by
    whichever query happened to run first.
    """
    base = settings.relevance_threshold if threshold is None else threshold

    best: dict[str, Product] = {}
    for i, (scores, indices) in enumerate(search_results):
        query_threshold = threshold_for(queries[i], base) if queries else base
        for product in filter_by_threshold(products, scores, indices, query_threshold):
            existing = best.get(product.asin)
            if existing is None or (product.score or 0) > (existing.score or 0):
                best[product.asin] = product

    ranked = sorted(best.values(), key=lambda p: p.score or 0, reverse=True)
    logger.info(
        "Evaluator kept %d/%d products at base threshold %.2f",
        len(ranked), len(products), base,
    )
    return ranked
