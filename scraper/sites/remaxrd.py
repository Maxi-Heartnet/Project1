"""remaxrd.com site parser.

Scrapes real estate listings from remaxrd.com using Playwright (Next.js, JS-rendered).
Returns a list of dicts with 6 keys: price, sector, property_type, bedrooms, area_m2, source_url.
Non-USD listings and field-incomplete listings are skipped.

Selectors spiked on 2026-04-10. remaxrd is single-page — URL params ?page=2 return the
same 16 cards, so pagination stops after page 1.

All selectors use stable semantic attributes (href patterns, img src patterns) rather than
CSS-in-JS class names (sc-*, css-*) which change across deployments.
"""
import logging
import re

from bs4 import BeautifulSoup

from scraper.scraper import parse_price_usd

logger = logging.getLogger(__name__)

REMAXRD_BASE = 'https://www.remaxrd.com'
LISTING_URL = 'https://www.remaxrd.com/propiedades'

CARD_SELECTOR = 'a[href*="/propiedad/"]'


def _parse_bedrooms(text):
    """Return bedroom count from text, or None."""
    m = re.search(r'(\d+)', text)
    return int(m.group(1)) if m else None


def _parse_area(text):
    """Return area in m2 from text containing M2/m2/mt2 indicator, or None."""
    m = re.search(r'(\d+(?:[.,]\d+)?)\s*(?:m[²2]|mt2)', text, re.IGNORECASE)
    if m:
        return float(re.sub(r',', '.', m.group(1)))
    return None


def _infer_property_type(text):
    """Infer property type from listing title text; default to 'apartment'."""
    lower = text.lower()
    if 'casa' in lower or 'house' in lower or 'villa' in lower:
        return 'house'
    return 'apartment'


def _parse_listings(html: str) -> list:
    """Parse rendered HTML and return listing dicts. Separated for testability."""
    soup = BeautifulSoup(html, 'html.parser')
    results = []
    skipped = 0

    # Cards are anchor tags linking to individual property pages
    seen_hrefs = set()
    for card in soup.select(CARD_SELECTOR):
        href = card.get('href', '')
        # Deduplicate — same card may appear multiple times in DOM
        if href in seen_hrefs:
            continue
        seen_hrefs.add(href)

        source_url = href if href.startswith('http') else REMAXRD_BASE + href

        # Price — find span containing "US$"
        price = None
        for span in card.find_all('span'):
            text = span.get_text(strip=True)
            if 'US$' in text or 'USD' in text.upper():
                price = parse_price_usd(text)
                if price is not None:
                    break
        if price is None:
            skipped += 1
            logger.debug('remaxrd: non-USD or missing price at %s, skipping', source_url)
            continue

        # Property type from h3
        h3 = card.find('h3')
        title_text = h3.get_text(strip=True) if h3 else ''
        property_type = _infer_property_type(title_text)

        # Sector — span sibling immediately after h3, split on comma, take first segment
        sector = None
        if h3:
            sib = h3.find_next_sibling('span')
            if sib:
                raw = sib.get_text(strip=True)
                parts = [p.strip() for p in raw.split(',') if p.strip()]
                if parts:
                    sector = parts[0].title()
        if not sector:
            skipped += 1
            logger.debug('remaxrd: missing sector at %s, skipping', source_url)
            continue

        # Bedrooms — li containing bed icon img
        bedrooms = None
        for li in card.find_all('li'):
            img = li.find('img', src=re.compile(r'icon_bed', re.IGNORECASE))
            if img:
                bedrooms = _parse_bedrooms(li.get_text(strip=True))
                break
        if bedrooms is None:
            skipped += 1
            logger.debug('remaxrd: missing bedrooms at %s, skipping', source_url)
            continue

        # Area — li containing rule/area icon img
        area_m2 = None
        for li in card.find_all('li'):
            img = li.find('img', src=re.compile(r'icon_rule', re.IGNORECASE))
            if img:
                area_m2 = _parse_area(li.get_text(strip=True))
                break
        if area_m2 is None:
            skipped += 1
            logger.debug('remaxrd: missing area at %s, skipping', source_url)
            continue

        results.append({
            'price': price,
            'sector': sector,
            'property_type': property_type,
            'bedrooms': bedrooms,
            'area_m2': area_m2,
            'source_url': source_url,
        })

    if skipped:
        logger.debug('remaxrd: skipped %d listings (non-USD or incomplete)', skipped)

    return results


def scrape(browser=None) -> list:
    """Scrape remaxrd.com and return a list of listing dicts.

    remaxrd is single-page — pagination stops after one fetch.
    Accepts an optional Playwright browser instance for testing; creates one if not provided.
    """
    from playwright.sync_api import sync_playwright

    def _fetch(browser_instance):
        page = browser_instance.new_page()
        try:
            page.goto(LISTING_URL, wait_until='networkidle', timeout=30_000)
            return page.content()
        finally:
            page.close()

    if browser is not None:
        html = _fetch(browser)
        return _parse_listings(html)

    with sync_playwright() as p:
        browser_instance = p.chromium.launch(headless=True)
        try:
            html = _fetch(browser_instance)
        finally:
            browser_instance.close()

    return _parse_listings(html)
