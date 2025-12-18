from __future__ import annotations

import re
from typing import Any

from bs4 import BeautifulSoup


_RATING_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)")
_HELPFUL_RE = re.compile(r"(\d[\d,]*)")


def _first_text(el) -> str:
    if not el:
        return ""
    return el.get_text(" ", strip=True)


def _parse_rating(text: str) -> float:
    m = _RATING_RE.search(text or "")
    if not m:
        return 0.0
    try:
        return float(m.group(1))
    except Exception:
        return 0.0


def _parse_helpful_votes(text: str) -> int:
    m = _HELPFUL_RE.search(text or "")
    if not m:
        return 0
    try:
        return int(m.group(1).replace(",", ""))
    except Exception:
        return 0


def _clean_review_date(text: str) -> str:
    t = (text or "").strip()
    # Common pattern: "Reviewed in the United States on January 1, 2024"
    t = t.replace("Reviewed in the United States on", "").strip()
    if " on " in t and t.lower().startswith("reviewed in"):
        t = t.split(" on ", 1)[1].strip()
    return t


def parse_amazon_reviews_html(html: str) -> list[dict[str, Any]]:
    """
    Extracts reviews from an Amazon "product-reviews" HTML page saved from a browser.
    Returns list of dicts compatible with the app's review schema.
    """
    soup = BeautifulSoup(html or "", "lxml")
    blocks = soup.select("li[data-hook='review'], div[data-hook='review']")

    reviews: list[dict[str, Any]] = []
    for i, b in enumerate(blocks, start=1):
        rid = (b.get("id") or "").strip() or f"R{i}"

        title_anchor = b.select_one("a[data-hook='review-title']") or b.select_one("[data-hook='review-title']")
        title = ""
        if title_anchor:
            spans = title_anchor.select("span")
            if spans:
                title = spans[-1].get_text(" ", strip=True)
            else:
                title = _first_text(title_anchor)
        body = _first_text(b.select_one("span[data-hook='review-body']"))

        rating_txt = _first_text(b.select_one("i[data-hook='review-star-rating'] span")) or _first_text(
            b.select_one("i[data-hook='cmps-review-star-rating'] span")
        )
        rating = _parse_rating(rating_txt)

        date_txt = _first_text(b.select_one("span[data-hook='review-date']"))
        date_txt = _clean_review_date(date_txt)

        verified = bool(b.select_one("[data-hook='avp-badge']"))

        helpful_txt = _first_text(b.select_one("span[data-hook='helpful-vote-statement']"))
        helpful_votes = _parse_helpful_votes(helpful_txt)

        reviews.append(
            {
                "review_id": rid,
                "title": title,
                "body": body,
                "rating": rating,
                "date": date_txt,
                "verified": verified,
                "helpful_votes": helpful_votes,
            }
        )

    return reviews
