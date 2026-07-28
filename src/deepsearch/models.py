"""Core data types.

Deliberately dependency-free: every other module can import this without pulling
in cohere, torch, or faiss. That keeps the domain logic (and its tests) cheap.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Product:
    """A single marketplace listing, plus the relevance score it earned."""

    keyword: str
    asin: str
    title: str | None = None
    description: str | None = None
    category: str | None = None
    price: str | None = None
    rating: str | None = None
    reviews: str | None = None
    url: str | None = None
    score: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)
