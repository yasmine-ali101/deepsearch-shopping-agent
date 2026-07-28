"""Dense retrieval over scraped products with FAISS.

Embeddings come from BAAI/bge-m3, chosen because it handles Arabic and English
in one shared vector space — the Arabic half of Agent 1's rewrites would be
useless against an English-only encoder.
"""

from __future__ import annotations

import functools
import logging

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import settings
from ..models import Product

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def get_encoder(model_name: str | None = None) -> SentenceTransformer:
    """Load (and cache) the sentence encoder. First call downloads ~2GB."""
    name = model_name or settings.embedding_model
    logger.info("Loading embedding model %s", name)
    return SentenceTransformer(name)


def build_product_text(product: Product) -> str:
    """Flatten a product into the single string that gets embedded."""
    parts = [product.title or "", product.description or ""]
    if product.category:
        parts.append(f"Category: {product.category}")
    return " | ".join(p for p in parts if p)


class ProductIndex:
    """A FAISS inner-product index over L2-normalised product embeddings.

    Vectors are normalised at encode time, so inner product == cosine similarity
    and scores land in a readable [-1, 1] range that the evaluator can threshold.
    """

    def __init__(self, products: list[Product], encoder: SentenceTransformer | None = None) -> None:
        if not products:
            raise ValueError("Cannot build an index over zero products.")

        self.products = products
        self._encoder = encoder or get_encoder()

        texts = [build_product_text(p) for p in products]
        embeddings = self._encoder.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)

    def search(self, queries: list[str], k: int | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return `(scores, indices)` per query, in the order given."""
        k = k or settings.top_k_per_query
        k = min(k, len(self.products))
        query_embeddings = self._encoder.encode(
            queries, convert_to_numpy=True, normalize_embeddings=True
        )
        scores, indices = self._index.search(query_embeddings, k)
        return list(zip(scores, indices))
