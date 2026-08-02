"""Agent 1, Query Rewriter.

Turns one messy shopping query into six focused retrieval queries (3 English,
3 Arabic), then strips filler words that hurt keyword search.

Bilingual expansion is the point: the target marketplaces (amazon.eg, amazon.sa,
amazon.ae) carry listings in both languages, and an English-only query silently
misses the Arabic half of the catalogue.
"""

from __future__ import annotations

import json
import logging

import cohere

from ..config import settings

logger = logging.getLogger(__name__)

_EXPANSION_PROMPT = """\
Return only valid JSON. Do not include explanations or text outside JSON.

Task:
1. Paraphrase this query into exactly 3 different natural-language queries in English.
2. Paraphrase this query into exactly 3 different natural-language queries in Arabic.
   Make them semantically equivalent to the English queries, but not literal translations.

User Query: "{user_query}"

Return strictly in this JSON format:
{{
  "queries_en": ["english_query1", "english_query2", "english_query3"],
  "queries_ar": ["arabic_query1", "arabic_query2", "arabic_query3"]
}}
"""

_CLEANING_PROMPT = """\
Return only valid JSON. Do not include explanations or text outside JSON.

Task:
1. Take the following list of queries and remove non-useful words such as:
   - filler words ("please", "find", "show", "best way to", etc.)
   - stop words that don't affect product search
   - any country name ("Egypt", "America", "UAE", etc.)
2. Keep the query concise and semantically equivalent.

Queries:
{queries}

Return in this JSON format:
{{"cleaned_queries": ["cleaned_query1", "cleaned_query2"]}}
"""


def _parse_json_response(raw: str) -> dict:
    """Parse a model response that may be wrapped in a markdown code fence.

    Cohere honours the "JSON only" instruction most of the time but not always,
    so we retry once after stripping fences rather than failing the whole round.
    """
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        stripped = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(stripped)


class QueryRewriter:
    """Expands and cleans a user query into retrieval-ready search terms."""

    def __init__(self, client: cohere.Client | None = None) -> None:
        self._client = client or cohere.Client(settings.cohere_api_key())

    def expand(self, user_query: str) -> dict[str, list[str]]:
        """Return `{"queries_en": [...], "queries_ar": [...]}`."""
        response = self._client.chat(
            model=settings.cohere_model,
            message=_EXPANSION_PROMPT.format(user_query=user_query),
            temperature=settings.expansion_temperature,
        )
        try:
            data = _parse_json_response(response.text)
        except json.JSONDecodeError:
            logger.warning("Query expansion returned unparseable JSON; falling back to raw query")
            return {"queries_en": [user_query], "queries_ar": []}
        return data

    def clean(self, queries: list[str]) -> list[str]:
        """Strip filler and geography from queries so keyword search stays tight."""
        listing = "\n".join(f"- {q}" for q in queries)
        response = self._client.chat(
            model=settings.cohere_model,
            message=_CLEANING_PROMPT.format(queries=listing),
            temperature=settings.cleaning_temperature,
        )
        try:
            return _parse_json_response(response.text)["cleaned_queries"]
        except (json.JSONDecodeError, KeyError):
            logger.warning("Query cleaning failed; using the expanded queries unchanged")
            return queries

    def rewrite(self, user_query: str) -> list[str]:
        """Full Agent 1 pass: expand into 6 queries, then clean them."""
        expanded = self.expand(user_query)
        all_queries = expanded.get("queries_en", []) + expanded.get("queries_ar", [])
        if not all_queries:
            return [user_query]
        return self.clean(all_queries)
