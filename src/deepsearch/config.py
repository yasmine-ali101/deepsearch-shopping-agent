"""Runtime configuration, loaded from environment variables.

No secret is ever hardcoded here. Copy `.env.example` to `.env` and fill it in.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


class MissingCredentialError(RuntimeError):
    """Raised when a required API key is not present in the environment."""


@dataclass(frozen=True)
class Settings:
    # --- Cohere (Agents 1 and 3) ---
    cohere_model: str = os.getenv("COHERE_MODEL", "command-r-plus")
    expansion_temperature: float = float(os.getenv("EXPANSION_TEMPERATURE", "0.8"))
    cleaning_temperature: float = float(os.getenv("CLEANING_TEMPERATURE", "0.5"))
    answer_temperature: float = float(os.getenv("ANSWER_TEMPERATURE", "0.6"))

    # --- Embeddings / retrieval ---
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    relevance_threshold: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.54"))
    top_k_per_query: int = int(os.getenv("TOP_K_PER_QUERY", "3"))

    # Arabic queries score systematically lower than semantically identical
    # English ones when the catalogue text is English, because the shared
    # embedding space is not perfectly isotropic across scripts. Measured on
    # 10 matched query pairs against the same products: mean 0.678 English
    # against 0.571 Arabic, a gap of 0.107. A single fixed threshold therefore
    # discards the Arabic rewrites at a much higher rate (at 0.54, English
    # passes 10/10 and Arabic 7/10), silently defeating the bilingual expansion.
    #
    # Re-measure for your own catalogue with scripts/calibrate_threshold.py.
    arabic_threshold_offset: float = float(os.getenv("ARABIC_THRESHOLD_OFFSET", "0.11"))

    # --- Scraper ---
    amazon_country: str = os.getenv("AMAZON_COUNTRY", "eg")
    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "15"))
    scrape_delay_seconds: float = float(os.getenv("SCRAPE_DELAY_SECONDS", "1.5"))
    user_agent: str = os.getenv(
        "USER_AGENT",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36",
    )

    # --- Orchestration ---
    max_rounds: int = int(os.getenv("MAX_ROUNDS", "3"))

    def cohere_api_key(self) -> str:
        key = os.getenv("COHERE_API_KEY")
        if not key:
            raise MissingCredentialError(
                "COHERE_API_KEY is not set. Copy .env.example to .env and add your key, "
                "or export it in your shell."
            )
        return key

    @property
    def scrape_headers(self) -> dict[str, str]:
        return {"User-Agent": self.user_agent, "Accept-Language": "en-US,en;q=0.9"}


settings = Settings()
