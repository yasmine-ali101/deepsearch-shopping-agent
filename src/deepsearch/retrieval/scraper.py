"""Amazon search-results scraper.

Scope note: this is a *research* retriever built for a course project. It reads
public search pages only, rate-limits itself, and gives up politely when Amazon
serves a CAPTCHA rather than trying to work around it. Amazon's Terms of Service
prohibit scraping; for anything beyond coursework use the Product Advertising API
or a licensed data provider. See the "Limitations" section of the README.
"""

from __future__ import annotations

import logging
import time

import requests
from bs4 import BeautifulSoup

from ..config import settings
from ..models import Product

logger = logging.getLogger(__name__)


class BlockedError(RuntimeError):
    """Raised when Amazon serves a CAPTCHA or robot check instead of results."""


def _get(url: str, params: dict | None = None) -> requests.Response:
    response = requests.get(
        url,
        params=params,
        headers=settings.scrape_headers,
        timeout=settings.request_timeout,
    )
    body = response.text.lower()
    if any(marker in body for marker in ("captcha", "enter the characters", "are you a human")):
        raise BlockedError("Blocked by Amazon (captcha/robot check detected).")
    return response


def fetch_description_and_category(url: str) -> tuple[str | None, str | None]:
    """Open a product page and pull its feature bullets and breadcrumb trail.

    Both fields feed the embedding text, so a miss here degrades ranking quality
    but must never abort the crawl, hence the broad except.
    """
    try:
        response = _get(url)
    except (requests.RequestException, BlockedError) as exc:
        logger.debug("Could not fetch product page %s: %s", url, exc)
        return None, None

    if response.status_code != 200:
        return None, None

    soup = BeautifulSoup(response.text, "html.parser")

    description = None
    bullets = soup.select("#feature-bullets ul li span")
    if bullets:
        description = " ".join(b.get_text(strip=True) for b in bullets if b.get_text(strip=True))
    else:
        element = soup.select_one("#productDescription")
        if element:
            description = element.get_text(strip=True)

    category = None
    crumbs = soup.select("#wayfinding-breadcrumbs_feature_div ul li a")
    if crumbs:
        category = " > ".join(a.get_text(strip=True) for a in crumbs if a.get_text(strip=True))

    return description, category


def _extract_price(item) -> str | None:
    whole = item.select_one("span.a-price-whole")
    if not whole:
        return None
    price = whole.get_text(strip=True)
    fraction = item.select_one("span.a-price-fraction")
    return f"{price}.{fraction.get_text(strip=True)}" if fraction else price


def _extract_rating(item) -> str | None:
    element = item.select_one("span.a-icon-alt")
    if element and "out of" in element.get_text():
        return element.get_text(strip=True).split(" out of")[0]
    return None


def search(keyword: str, k: int = 5, country: str | None = None) -> list[Product]:
    """Scrape the first page of Amazon search results for `keyword`."""
    country = country or settings.amazon_country
    response = _get(f"https://www.amazon.{country}/s", params={"k": keyword})
    if response.status_code != 200:
        raise RuntimeError(f"Bad status {response.status_code} for URL: {response.url}")

    soup = BeautifulSoup(response.text, "html.parser")
    # Amazon rotates its result-card markup; the data-asin attribute is the one
    # stable anchor across layouts.
    items = soup.select("div.s-result-item[data-asin]") or soup.select("div[data-asin]")

    products: list[Product] = []
    for item in items:
        asin = item.get("data-asin", "").strip()
        if not asin:
            continue

        title_el = (
            item.select_one("h2 a span")
            or item.select_one("span.a-size-medium.a-color-base.a-text-normal")
            or item.select_one("a.a-link-normal.a-text-normal span")
            or item.select_one("h2")
        )
        link_el = item.select_one("h2 a") or item.select_one("a.a-link-normal.s-no-outline")

        url = None
        if link_el and link_el.get("href"):
            href = link_el["href"]
            url = href if href.startswith("http") else f"https://www.amazon.{country}{href}"

        description, category = fetch_description_and_category(url) if url else (None, None)
        reviews_el = item.select_one("span.a-size-base.s-underline-text")

        products.append(
            Product(
                keyword=keyword,
                asin=asin,
                title=title_el.get_text(strip=True) if title_el else None,
                description=description,
                category=category,
                price=_extract_price(item),
                rating=_extract_rating(item),
                reviews=reviews_el.get_text(strip=True).replace(",", "") if reviews_el else None,
                url=url,
            )
        )
        if len(products) >= k:
            break

    return products


def multi_search(keywords: list[str], k: int = 5, country: str | None = None) -> list[Product]:
    """Run `search` across every rewritten query, deduplicating by ASIN.

    One failed keyword must not sink the round, a CAPTCHA on query 4 of 6 still
    leaves five usable result sets.
    """
    collected: list[Product] = []
    seen: set[str] = set()

    for i, keyword in enumerate(keywords):
        if i:
            time.sleep(settings.scrape_delay_seconds)
        try:
            for product in search(keyword, k=k, country=country):
                if product.asin not in seen:
                    seen.add(product.asin)
                    collected.append(product)
        except (requests.RequestException, BlockedError, RuntimeError) as exc:
            logger.warning("Search failed for %r: %s", keyword, exc)

    return collected
