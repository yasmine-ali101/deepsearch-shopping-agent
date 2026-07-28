"""Gradio front-end.

Run with:  python -m deepsearch.app
"""

from __future__ import annotations

import logging

import gradio as gr

from .config import MissingCredentialError, settings
from .pipeline import ShoppingPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def _build_pipeline() -> ShoppingPipeline | None:
    try:
        return ShoppingPipeline()
    except MissingCredentialError as exc:
        logging.error("%s", exc)
        return None


def build_ui() -> gr.Blocks:
    pipeline = _build_pipeline()

    def on_search(user_query: str, country: str, threshold: float, max_rounds: float):
        if pipeline is None:
            return "**Configuration error** — `COHERE_API_KEY` is not set. Copy `.env.example` to `.env` and add your key."
        if not user_query.strip():
            return "Please enter a shopping query."
        try:
            result = pipeline.run(
                user_query=user_query,
                country=country,
                threshold=threshold,
                max_rounds=int(max_rounds),
            )
        except Exception as exc:  # noqa: BLE001 - surface errors in the UI
            logging.exception("Pipeline failed")
            return f"**Error:** {exc}"

        header = (
            f"_{len(result.products)} product(s) after {result.rounds_used} "
            f"round(s) of retrieval._\n\n"
        )
        return header + result.answer

    with gr.Blocks(theme="soft", title="Deep Search Shopping Assistant") as demo:
        gr.Markdown(
            "## Multi-Agent Shopping Assistant\n"
            "Bilingual (EN/AR) query expansion, live retrieval, relevance gating, "
            "and grounded answer generation."
        )

        user_query = gr.Textbox(
            label="Shopping query",
            placeholder="e.g. budget-friendly earbuds for travel",
            lines=2,
        )

        with gr.Row():
            country = gr.Dropdown(
                choices=["eg", "sa", "ae", "com", "co.uk"],
                value=settings.amazon_country,
                label="Amazon marketplace",
            )
            threshold = gr.Slider(
                0.30, 0.90, value=settings.relevance_threshold, step=0.01,
                label="Relevance threshold",
            )
            max_rounds = gr.Slider(
                1, 5, value=settings.max_rounds, step=1, label="Max retrieval rounds"
            )

        run_btn = gr.Button("Search", variant="primary")
        output = gr.Markdown(label="Result")

        run_btn.click(
            fn=on_search,
            inputs=[user_query, country, threshold, max_rounds],
            outputs=output,
        )
        gr.Examples(
            examples=[
                ["cheap laptop for school", "eg", 0.54, 3],
                ["noise cancelling headphones under 2000 EGP", "eg", 0.54, 3],
                ["مكنسة كهربائية صغيرة للسيارة", "eg", 0.50, 3],
            ],
            inputs=[user_query, country, threshold, max_rounds],
        )

    return demo


def main() -> None:
    build_ui().launch()


if __name__ == "__main__":
    main()
