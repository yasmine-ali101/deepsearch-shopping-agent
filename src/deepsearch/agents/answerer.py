"""Agent 3 — Answer Generator.

Takes the products Agent 2 approved and writes the user-facing reply. The
deterministic `format_products` summary is built first and passed to the model as
grounding, so the LLM is rewriting facts rather than inventing them — and if the
Cohere call fails, that summary is still a usable answer on its own.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import cohere

from ..config import settings
from ..models import Product

logger = logging.getLogger(__name__)

_ANSWER_PROMPT = """\
You are an AI shopping assistant.

Task:
1. Take the user query and the raw product results.
2. Summarize them into a clear and engaging final answer.
3. Keep it concise, helpful, and structured.
4. Only mention products present in the results. Do not invent products, prices,
   or ratings. If a field is missing, say so rather than guessing.

User Query: "{user_query}"

Products Found:
{products_json}

Base Summary:
{base_summary}

Return only the polished final answer in markdown.
"""


class Answerer:
    """Grounded answer generation with a conversation history."""

    def __init__(self, client: cohere.Client | None = None) -> None:
        self._client = client or cohere.Client(settings.cohere_api_key())
        self.history: list[dict] = []

    @staticmethod
    def format_products(products: list[Product]) -> str:
        """Render an approved product list as deterministic markdown."""
        if not products:
            return "Sorry, I couldn't find relevant products for that query."

        lines = ["Here are the top products I found:", ""]
        for i, p in enumerate(products, 1):
            lines.append(f"{i}. **{p.title or 'Unknown title'}**")
            lines.append(f"   - Price: {p.price or 'N/A'}")
            lines.append(f"   - Rating: {p.rating or 'N/A'} ({p.reviews or '0'} reviews)")
            if p.url:
                lines.append(f"   - [View on Amazon]({p.url})")
            if p.description:
                lines.append(f"   - {p.description[:200]}...")
            lines.append("")
        return "\n".join(lines)

    def answer(self, user_query: str, products: list[Product]) -> str:
        base_summary = self.format_products(products)
        payload = [p.as_dict() for p in products]

        self.history.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_query": user_query,
                "products": payload,
            }
        )

        if not products:
            return base_summary

        try:
            response = self._client.chat(
                model=settings.cohere_model,
                message=_ANSWER_PROMPT.format(
                    user_query=user_query,
                    products_json=json.dumps(payload, indent=2, ensure_ascii=False),
                    base_summary=base_summary,
                ),
                temperature=settings.answer_temperature,
            )
            answer = response.text.strip()
        except Exception as exc:  # noqa: BLE001 - degrade to the grounded summary
            logger.warning("Answer polishing failed (%s); returning base summary", exc)
            answer = base_summary

        self.history[-1]["answer"] = answer
        return answer
