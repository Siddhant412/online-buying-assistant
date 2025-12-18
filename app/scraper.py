"""
Amazon scraping connector (scaffolding).

Provides:
  - search_amazon(keyword) -> list of candidates
  - fetch_product_page(asin, url) -> raw HTML
  - fetch_reviews(asin, max_pages, sort) -> list of review dicts

Implementation note:
  This is a thin wrapper around Playwright. Network calls are orchestrated
  elsewhere; here we only outline the logic and normalize outputs.
"""
from __future__ import annotations
import asyncio
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional

# Playwright imports are optional; delay to runtime

USER_AGENT = os.getenv("SCRAPE_USER_AGENT", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
MAX_PAGES = int(os.getenv("SCRAPE_MAX_PAGES", "3"))
DELAY_SEC = float(os.getenv("SCRAPE_DELAY_SEC", "2.0"))
TIMEOUT_MS = int(os.getenv("SCRAPE_TIMEOUT_MS", "15000"))
HEADLESS = os.getenv("SCRAPE_HEADLESS", "1").strip().lower() not in {"0", "false", "no"}
SCRAPE_DEBUG = os.getenv("SCRAPE_DEBUG", "").strip().lower() in {"1", "true", "yes"}
SCRAPE_DEBUG_DIR = Path(os.getenv("SCRAPE_DEBUG_DIR", str(Path(__file__).resolve().parents[1] / "data" / "raw")))
PROFILE_DIR = Path(
    os.getenv(
        "SCRAPE_PROFILE_DIR",
        str(Path(__file__).resolve().parents[1] / ".playwright" / "profile"),
    )
)
USE_PERSISTENT_PROFILE = os.getenv("SCRAPE_PERSISTENT", "1").strip().lower() not in {"0", "false", "no"}
INTERACTIVE_LOGIN_TIMEOUT_SEC = int(os.getenv("SCRAPE_INTERACTIVE_LOGIN_TIMEOUT_SEC", "180"))

SEARCH_URL_TMPL = "https://www.amazon.com/s?k={query}"
REVIEWS_URL_TMPL = "https://www.amazon.com/product-reviews/{asin}/ref=cm_cr_dp_d_show_all_btm?sortBy={sort}&pageNumber={page}"

@dataclass
class SearchResult:
    asin: str
    title: str
    url: str
    price: Optional[str] = None

@dataclass
class Review:
    review_id: str
    title: str
    body: str
    rating: float
    date: str
    verified: bool
    helpful_votes: int


_BLOCK_PATTERNS = [
    "captchacharacters",
    "/errors/validatecaptcha",
    "robot check",
    "enter the characters you see below",
    "sorry, we just need to make sure you're not a robot",
    "to discuss automated access to amazon data",
    "unusual traffic",
]


def _detect_block(html: str) -> str | None:
    h = (html or "").lower()
    for p in _BLOCK_PATTERNS:
        if p in h:
            return p
    return None


def _maybe_write_debug(name: str, html: str) -> None:
    if not SCRAPE_DEBUG:
        return
    try:
        SCRAPE_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        (SCRAPE_DEBUG_DIR / name).write_text(html or "", encoding="utf-8", errors="ignore")
    except Exception:
        # best-effort only
        pass


async def _new_context(playwright):
    chromium = playwright.chromium
    if USE_PERSISTENT_PROFILE:
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        context = await chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=HEADLESS,
            locale="en-US",
            user_agent=USER_AGENT,
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
            viewport={"width": 1280, "height": 800},
        )
        return None, context

    browser = await chromium.launch(headless=HEADLESS)
    context = await browser.new_context(
        user_agent=USER_AGENT,
        locale="en-US",
        extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
        viewport={"width": 1280, "height": 800},
    )
    return browser, context


def _get_async_playwright():
    try:
        from playwright.async_api import async_playwright
    except ModuleNotFoundError as e:
        exe = sys.executable
        raise RuntimeError(
            "Playwright is not installed in the Python environment running this server.\n"
            f"Install it with:\n  {exe} -m pip install playwright\n"
            "Then install browser binaries with:\n"
            f"  {exe} -m playwright install chromium"
        ) from e
    return async_playwright


async def _maybe_wait_for_manual_login(page) -> None:
    """
    In headful mode, Amazon may redirect to /ap/signin. Give the user time to sign in,
    then continue once reviews appear.
    """
    if HEADLESS:
        return

    url = (getattr(page, "url", "") or "").lower()
    if "/ap/signin" not in url and "signin" not in url:
        return

    deadline = time.time() + max(5, INTERACTIVE_LOGIN_TIMEOUT_SEC)
    while time.time() < deadline:
        try:
            await page.wait_for_selector("li[data-hook='review'], div[data-hook='review']", timeout=1000)
            return
        except Exception:
            await page.wait_for_timeout(250)

    raise RuntimeError(
        "Timed out waiting for manual Amazon sign-in. "
        "Increase `SCRAPE_INTERACTIVE_LOGIN_TIMEOUT_SEC` and retry."
    )


async def _search_once(page, keyword: str, limit: int = 5) -> List[SearchResult]:
    await page.goto(SEARCH_URL_TMPL.format(query=keyword.replace(" ", "+")), timeout=TIMEOUT_MS)
    await page.wait_for_load_state("domcontentloaded")
    await page.wait_for_timeout(800)

    items: List[SearchResult] = []
    cards = await page.query_selector_all("div[data-asin][data-component-type='s-search-result']")
    for c in cards:
        asin = (await c.get_attribute("data-asin")) or ""
        if not asin:
            continue
        title_el = await c.query_selector("h2 a span")
        title = (await title_el.inner_text()) if title_el else asin
        link_el = await c.query_selector("h2 a")
        href = (await link_el.get_attribute("href")) if link_el else ""
        url = f"https://www.amazon.com{href.split('?')[0]}" if href else f"https://www.amazon.com/dp/{asin}"
        price_el = await c.query_selector("span.a-price > span.a-offscreen")
        price = (await price_el.inner_text()) if price_el else None
        items.append(SearchResult(asin=asin, title=title.strip(), url=url, price=price))
        if len(items) >= limit:
            break
    return items


async def _fetch_reviews_once(page, asin: str, max_pages: int, sort: str) -> List[Review]:
    sort_key = "recent" if sort not in ("recent", "helpful") else sort
    reviews: List[Review] = []

    for page_num in range(1, max_pages + 1):
        url = REVIEWS_URL_TMPL.format(asin=asin, sort=sort_key, page=page_num)
        await page.goto(url, timeout=TIMEOUT_MS)
        await page.wait_for_load_state("domcontentloaded")
        await _maybe_wait_for_manual_login(page)
        await page.wait_for_timeout(int(DELAY_SEC * 1000))

        try:
            await page.wait_for_selector(
                "li[data-hook='review'], div[data-hook='review']",
                timeout=min(7000, TIMEOUT_MS),
            )
        except Exception:
            pass

        html = await page.content() or ""
        _maybe_write_debug(f"{asin}_reviews_p{page_num}.html", html)

        blocks = await page.query_selector_all("li[data-hook='review'], div[data-hook='review']")
        if not blocks:
            reason = _detect_block(html)
            if reason:
                raise RuntimeError(
                    "Amazon returned a bot-check / blocked page while fetching reviews "
                    f"(matched: {reason}). Try `SCRAPE_HEADLESS=0` and sign in, or use a different network/location."
                )
            break

        for b in blocks:
            rid = (await b.get_attribute("id")) or ""
            title_el = await b.query_selector("a[data-hook='review-title']")
            title = ""
            if title_el:
                spans = await title_el.query_selector_all("span")
                if spans:
                    title = (await spans[-1].inner_text()) or ""
                else:
                    title = (await title_el.inner_text()) or ""
            title = title.strip()
            body_el = await b.query_selector("span[data-hook='review-body']")
            body = (await body_el.inner_text()) if body_el else ""
            rating_el = await b.query_selector("i[data-hook='review-star-rating'] span")
            rating_txt = (await rating_el.inner_text()) if rating_el else "0"
            try:
                rating = float(rating_txt.split()[0])
            except Exception:
                rating = 0.0
            date_el = await b.query_selector("span[data-hook='review-date']")
            date_txt = (await date_el.inner_text()) if date_el else ""
            verified_el = await b.query_selector("span.a-declarative span[data-hook='avp-badge']")
            verified = "Verified" in ((await verified_el.inner_text()) if verified_el else "")
            helpful_el = await b.query_selector("span[data-hook='helpful-vote-statement']")
            helpful_txt = (await helpful_el.inner_text()) if helpful_el else "0"
            try:
                hv = int(helpful_txt.split()[0].replace(",", ""))
            except Exception:
                hv = 0

            reviews.append(
                Review(
                    review_id=rid,
                    title=title.strip(),
                    body=body.strip(),
                    rating=rating,
                    date=date_txt.replace("Reviewed in the United States on", "").strip(),
                    verified=verified,
                    helpful_votes=hv,
                )
            )

        await asyncio.sleep(DELAY_SEC)

    return reviews


async def search_amazon(keyword: str, limit: int = 5) -> List[SearchResult]:
    async_playwright = _get_async_playwright()
    async with async_playwright() as p:
        browser, ctx = await _new_context(p)
        try:
            page = (ctx.pages[0] if ctx.pages else await ctx.new_page())
            return await _search_once(page, keyword, limit=limit)
        finally:
            await ctx.close()
            if browser:
                await browser.close()


async def fetch_product_page(asin: str, url: Optional[str] = None) -> str:
    async_playwright = _get_async_playwright()
    page_url = url or f"https://www.amazon.com/dp/{asin}"
    async with async_playwright() as p:
        browser, ctx = await _new_context(p)
        try:
            page = (ctx.pages[0] if ctx.pages else await ctx.new_page())
            await page.goto(page_url, timeout=TIMEOUT_MS)
            await page.wait_for_load_state("domcontentloaded")
            await page.wait_for_timeout(int(DELAY_SEC * 1000))
            html = await page.content() or ""
            _maybe_write_debug(f"{asin}_product.html", html)
            return html
        finally:
            await ctx.close()
            if browser:
                await browser.close()


async def fetch_reviews(asin: str, max_pages: int = MAX_PAGES, sort: str = "recent") -> List[Review]:
    async_playwright = _get_async_playwright()
    async with async_playwright() as p:
        browser, ctx = await _new_context(p)
        try:
            page = (ctx.pages[0] if ctx.pages else await ctx.new_page())
            return await _fetch_reviews_once(page, asin=asin, max_pages=max_pages, sort=sort)
        finally:
            await ctx.close()
            if browser:
                await browser.close()


async def scrape_amazon_async(keyword_or_url: str, max_pages: int = MAX_PAGES, sort: str = "recent") -> Dict:
    """
    Single Playwright session scrape (search -> product page -> reviews).
    Uses a persistent profile by default so your login persists across runs.
    """
    async_playwright = _get_async_playwright()

    async with async_playwright() as p:
        browser, ctx = await _new_context(p)
        try:
            page = (ctx.pages[0] if ctx.pages else await ctx.new_page())

            # Determine ASIN + URL
            if "amazon.com" in keyword_or_url and "/dp/" in keyword_or_url:
                asin = keyword_or_url.split("/dp/")[1].split("/")[0].split("?")[0]
                url = keyword_or_url
            else:
                results = await _search_once(page, keyword_or_url, limit=1)
                if not results:
                    raise RuntimeError("No results found")
                asin = results[0].asin
                url = results[0].url or f"https://www.amazon.com/dp/{asin}"

            # Product page HTML
            await page.goto(url, timeout=TIMEOUT_MS)
            await page.wait_for_load_state("domcontentloaded")
            await page.wait_for_timeout(int(DELAY_SEC * 1000))
            html = await page.content() or ""
            _maybe_write_debug(f"{asin}_product.html", html)

            # Reviews
            revs = await _fetch_reviews_once(page, asin=asin, max_pages=max_pages, sort=sort)

            return {
                "asin": asin,
                "url": url,
                "html": html,
                "reviews": [asdict(r) for r in revs],
                "meta": {"sort": sort, "pages": max_pages},
            }
        finally:
            await ctx.close()
            if browser:
                await browser.close()


def scrape_amazon_sync(keyword_or_url: str, max_pages: int = MAX_PAGES, sort: str = "recent") -> Dict:
    """
    Synchronous wrapper for orchestrating search + product fetch + reviews.
    Returns a dict with product_id (asin), html, reviews[], meta.
    """
    async def _run():
        return await scrape_amazon_async(keyword_or_url, max_pages=max_pages, sort=sort)
    try:
        return asyncio.run(_run())
    except RuntimeError:
        # Fallback for environments where no event loop is set in the current thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_run())
        finally:
            loop.close()
            asyncio.set_event_loop(None)
