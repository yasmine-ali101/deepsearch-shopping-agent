"""Deep Search, a multi-agent RAG shopping assistant."""

from .models import Product
from .pipeline import PipelineResult, ShoppingPipeline, run_pipeline

__version__ = "0.1.0"
__all__ = ["ShoppingPipeline", "PipelineResult", "Product", "run_pipeline"]
