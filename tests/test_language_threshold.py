"""Tests for the language aware relevance threshold.

Arabic rewrites score systematically lower than equivalent English ones against
an English catalogue (measured gap 0.107). Judging both against one fixed bar
throws away the Arabic half of the retrieval, which is the whole point of the
bilingual expansion. These pin the correction.
"""

import numpy as np
import pytest

from deepsearch.agents.evaluator import evaluate, is_arabic, threshold_for
from deepsearch.config import settings
from deepsearch.models import Product


def products(n: int) -> list[Product]:
    return [Product(keyword="q", asin=f"A{i}", title=f"Product {i}") for i in range(n)]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("cheap laptop for school", False),
        ("لابتوب رخيص للمدرسة", True),
        ("سماعات عازلة للضوضاء", True),
        ("", False),
        ("12345", False),
        ("!!!", False),
    ],
)
def test_script_detection(text, expected):
    assert is_arabic(text) is expected


def test_mostly_english_query_naming_an_arabic_brand_counts_as_english():
    """A majority test, not a 'contains any Arabic character' test."""
    assert is_arabic("wireless headphones from سوني store") is False


def test_arabic_queries_get_a_lower_bar():
    base = 0.54

    assert threshold_for("cheap laptop", base) == pytest.approx(base)
    assert threshold_for("لابتوب رخيص", base) == pytest.approx(
        base - settings.arabic_threshold_offset
    )


def test_an_arabic_result_below_the_english_bar_is_still_kept():
    """The exact case the fix exists for.

    0.53 fails a flat 0.54 threshold but clears the Arabic adjusted 0.43, so a
    genuinely relevant Arabic hit is no longer discarded.
    """
    catalogue = products(1)
    hits = [(np.array([0.53]), np.array([0]))]

    without_language = evaluate(catalogue, hits, threshold=0.54)
    with_language = evaluate(catalogue, hits, threshold=0.54, queries=["لابتوب رخيص للمدرسة"])

    assert without_language == []
    assert [p.asin for p in with_language] == ["A0"]


def test_the_english_bar_is_unchanged_by_the_fix():
    """Lowering the Arabic bar must not quietly lower the English one."""
    catalogue = products(1)
    hits = [(np.array([0.53]), np.array([0]))]

    result = evaluate(catalogue, hits, threshold=0.54, queries=["cheap laptop for school"])

    assert result == []


def test_a_weak_arabic_result_is_still_rejected():
    """The offset is a correction, not an amnesty."""
    catalogue = products(1)
    hits = [(np.array([0.20]), np.array([0]))]

    result = evaluate(catalogue, hits, threshold=0.54, queries=["لابتوب رخيص"])

    assert result == []


def test_mixed_language_queries_are_each_judged_on_their_own_terms():
    catalogue = products(2)
    hits = [
        (np.array([0.53]), np.array([0])),   # English query, should fail 0.54
        (np.array([0.53]), np.array([1])),   # Arabic query, should pass 0.43
    ]

    result = evaluate(
        catalogue, hits, threshold=0.54,
        queries=["cheap laptop", "لابتوب رخيص"],
    )

    assert [p.asin for p in result] == ["A1"]


def test_omitting_queries_falls_back_to_a_single_threshold():
    """Backwards compatible: callers that pass no queries get the old behaviour."""
    catalogue = products(2)
    hits = [(np.array([0.60, 0.53]), np.array([0, 1]))]

    result = evaluate(catalogue, hits, threshold=0.54)

    assert [p.asin for p in result] == ["A0"]
