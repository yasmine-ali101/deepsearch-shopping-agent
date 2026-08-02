"""Measure the English/Arabic scoring gap for a given encoder and catalogue.

    python scripts/calibrate_threshold.py

Agent 1 rewrites every query into three English and three Arabic variants, and
all six are scored against the same index. If the encoder scores Arabic lower
against English catalogue text, a single fixed threshold discards the Arabic
rewrites at a higher rate and quietly defeats the bilingual design.

This script quantifies that gap on matched query pairs meaning the same thing,
and prints the offset to put in `ARABIC_THRESHOLD_OFFSET`. Needs no API key:
it exercises the encoder and index only.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deepsearch.config import settings  # noqa: E402
from deepsearch.models import Product  # noqa: E402
from deepsearch.retrieval.index import ProductIndex  # noqa: E402


def product(asin: str, title: str, description: str, category: str) -> Product:
    return Product(
        keyword="calibration",
        asin=asin,
        title=title,
        description=description,
        category=category,
    )


CATALOGUE = [
    product("A1", "HP 15 Laptop, Intel Celeron, 4GB RAM, 128GB SSD",
            "Budget laptop for students, lightweight, long battery life for school work",
            "Electronics > Computers > Laptops"),
    product("A2", "Lenovo IdeaPad Slim 3, Ryzen 5, 8GB RAM",
            "Affordable everyday laptop suitable for study and office tasks",
            "Electronics > Computers > Laptops"),
    product("A3", "Dell XPS 15 Professional Workstation, i9, 32GB RAM",
            "High end premium laptop for video editing and 3D rendering",
            "Electronics > Computers > Laptops"),
    product("A4", "Sony WH-1000XM5 Noise Cancelling Headphones",
            "Wireless over ear headphones with industry leading noise cancellation",
            "Electronics > Audio > Headphones"),
    product("A5", "Anker Soundcore Life Q30 Headphones",
            "Budget friendly noise cancelling wireless headphones for travel",
            "Electronics > Audio > Headphones"),
    product("A6", "Portable Car Vacuum Cleaner 12V Handheld",
            "Small handheld vacuum for cleaning car interior, plugs into car socket",
            "Automotive > Car Care > Vacuums"),
    product("A7", "Stainless Steel Cooking Pot Set, 5 pieces",
            "Kitchen cookware set, induction compatible, dishwasher safe",
            "Home > Kitchen > Cookware"),
    product("A8", "Nike Air Zoom Running Shoes Men",
            "Lightweight running shoes with responsive cushioning for road running",
            "Fashion > Shoes > Athletic"),
    product("A9", "Samsung Galaxy A54 Smartphone 128GB",
            "Mid range smartphone with AMOLED display and 50MP camera",
            "Electronics > Mobile > Smartphones"),
    product("A10", "Baby Stroller Foldable Lightweight",
            "Compact foldable stroller for infants, one hand fold, travel friendly",
            "Baby > Strollers"),
]

# (english, arabic, asin the query should retrieve)
PAIRS = [
    ("cheap laptop for school", "لابتوب رخيص للمدرسة", "A1"),
    ("affordable laptop for study", "لابتوب مناسب للدراسة", "A2"),
    ("laptop for video editing", "لابتوب لتحرير الفيديو", "A3"),
    ("noise cancelling headphones", "سماعات عازلة للضوضاء", "A4"),
    ("budget wireless headphones", "سماعات لاسلكية رخيصة", "A5"),
    ("car vacuum cleaner", "مكنسة كهربائية للسيارة", "A6"),
    ("cooking pot set", "طقم حلل للطبخ", "A7"),
    ("running shoes for men", "حذاء جري رجالي", "A8"),
    ("smartphone with good camera", "موبايل بكاميرا كويسة", "A9"),
    ("foldable baby stroller", "عربة أطفال قابلة للطي", "A10"),
]


def main() -> None:
    print(f"Encoder: {settings.embedding_model}")
    print(f"Building index over {len(CATALOGUE)} products\n")
    index = ProductIndex(CATALOGUE)
    position = {p.asin: i for i, p in enumerate(CATALOGUE)}

    english, arabic = [], []
    print(f"{'english query':32s} {'en':>6s}  {'ar':>6s}  {'gap':>7s}")
    print("-" * 58)

    for english_query, arabic_query, asin in PAIRS:
        (scores_en, idx_en), = index.search([english_query], k=len(CATALOGUE))
        (scores_ar, idx_ar), = index.search([arabic_query], k=len(CATALOGUE))
        target = position[asin]
        score_en = float(scores_en[list(idx_en).index(target)])
        score_ar = float(scores_ar[list(idx_ar).index(target)])
        english.append(score_en)
        arabic.append(score_ar)
        print(f"{english_query:32s} {score_en:6.3f}  {score_ar:6.3f}  {score_ar - score_en:+7.3f}")

    mean_en = statistics.mean(english)
    mean_ar = statistics.mean(arabic)
    gap = mean_en - mean_ar

    print(f"\nmean similarity to the correct product:")
    print(f"  English {mean_en:.3f}")
    print(f"  Arabic  {mean_ar:.3f}")
    print(f"  gap     {gap:.3f}")

    print(f"\npass rates at a single fixed threshold:")
    for threshold in (0.50, 0.54, 0.58, 0.62):
        passed_en = sum(1 for s in english if s >= threshold)
        passed_ar = sum(1 for s in arabic if s >= threshold)
        print(f"  {threshold:.2f}: English {passed_en:2d}/{len(english)}   "
              f"Arabic {passed_ar:2d}/{len(arabic)}")

    print(f"\npass rates with the offset applied to Arabic:")
    for threshold in (0.50, 0.54, 0.58, 0.62):
        passed_en = sum(1 for s in english if s >= threshold)
        passed_ar = sum(1 for s in arabic if s >= threshold - gap)
        print(f"  {threshold:.2f}: English {passed_en:2d}/{len(english)}   "
              f"Arabic {passed_ar:2d}/{len(arabic)}")

    print(f"\nSuggested setting:\n  ARABIC_THRESHOLD_OFFSET={gap:.2f}")
    print(f"  (currently {settings.arabic_threshold_offset:.2f})")


if __name__ == "__main__":
    main()
