"""Retrieval components.

Both submodules pull in heavy third-party dependencies (requests/bs4, and
torch/faiss respectively), so they are exposed lazily.
"""

__all__ = ["Product", "BlockedError", "search", "multi_search", "ProductIndex", "get_encoder"]

_LAZY = {
    "BlockedError": ".scraper",
    "search": ".scraper",
    "multi_search": ".scraper",
    "ProductIndex": ".index",
    "build_product_text": ".index",
    "get_encoder": ".index",
}


def __getattr__(name: str):
    if name == "Product":
        from ..models import Product

        return Product
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module, __name__), name)
