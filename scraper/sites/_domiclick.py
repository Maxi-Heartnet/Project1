"""Shared parser for sites running on the Domiclick platform.

Both apartamentosrd.com.do and tucasard.com run on Domiclick and are structurally
identical. This helper is parameterised by base_url so each site module is a thin wrapper.

Spiked on 2026-04-10:
- Cards: div.card.h-100
- Anchor: a[href*="/propiedad/"] inside card
- Property type: h5 inside div.property-content
- Sector: span containing i.fas.fa-map-marker-alt — e.g. "Serrallés, Santo Domingo D.N."
- Price: span in price ul — "US$ 165,000" or "RD$ X,XXX,XXX"
- Bedrooms: li with i.fas.fa-bed — "1 Hab." or "Desde 1 hasta 3 Hab."
- Area: li with i.fas.fa-arrows-alt — "58 Mt2"
- Pagination: ?listing_type=1&page=N; stop when no div.card.h-100 on page
"""
import logging
import re

from bs4 import BeautifulSoup

from scraper.scraper import parse_price_usd

logger = logging.getLogger(__name__)

CARD_SELECTOR = 'div.card.h-100'
LISTING_URL_TMPL = '{base}/propiedades?listing_type=1&page={page}'


def _parse_bedrooms(text):
    """Return bedroom count from text; extracts first integer found."""
    m = re.search(r'(\d+)', text)
    return int(m.group(1)) if m else None


def _parse_area(text):
    """Return area in m2 from text containing m²/m2/mt2 indicator, or None."""
    m = re.search(r'(\d+(?:[.,]\d+)?)\s*(?:m[²2]|mt2)', text, re.IGNORECASE)
    if m:
        return float(re.sub(r',', '.', m.group(1)))
    return None


def _infer_property_type(text):
    """Infer property type from title text; default to 'apartment'."""
    lower = text.lower()
    if 'casa' in lower or 'house' in lower or 'villa' in lower:
        return 'house'
    return 'apartment'


def parse_listings(html: str, base_url: str) -> list:
    """Parse rendered HTML for one Domiclick page and return listing dicts.

    Args:
        html: Rendered page HTML (from Playwright page.content()).
        base_url: Site base URL, e.g. 'https://www.apartamentosrd.com.do'.

    Returns:
        List of listing dicts with 6 keys: price, sector, property_type,
        bedrooms, area_m2, source_url.
    """
    soup = BeautifulSoup(html, 'html.parser')
    results = []
    skipped = 0

    for card in soup.select(CARD_SELECTOR):
        # Source URL
        anchor = card.select_one('a[href*="/propiedad/"]')
        if not anchor:
            skipped += 1
            continue
        href = anchor['href']
        source_url = href if href.startswith('http') else base_url + href

        # Price
        price = None
        for span in card.find_all('span'):
            text = span.get_text(strip=True)
            if text:
                price = parse_price_usd(text)
                if price is not None:
                    break
        if price is None:
            skipped += 1
            logger.debug('domiclick(%s): non-USD or missing price at %s, skipping', base_url, source_url)
            continue

        # Property type from h5
        h5 = card.select_one('div.property-content h5')
        title_text = h5.get_text(strip=True) if h5 else ''
        property_type = _infer_property_type(title_text)

        # Sector — span with fa-map-marker-alt icon
        sector = None
        marker = card.find('i', class_='fa-map-marker-alt')
        if marker:
            span = marker.find_parent('span') or marker.find_next_sibling()
            if span is None:
                span = marker.parent
            raw = span.get_text(separator=' ', strip=True)
            parts = [p.strip() for p in raw.split(',') if p.strip()]
            if parts:
                sector = parts[0]
        if not sector:
            skipped += 1
            logger.debug('domiclick(%s): missing sector at %s, skipping', base_url, source_url)
            continue

        # Bedrooms — li with fa-bed icon
        bedrooms = None
        bed_icon = card.find('i', class_='fa-bed')
        if bed_icon:
            li = bed_icon.find_parent('li') or bed_icon.parent
            bedrooms = _parse_bedrooms(li.get_text(strip=True))
        if bedrooms is None:
            skipped += 1
            logger.debug('domiclick(%s): missing bedrooms at %s, skipping', base_url, source_url)
            continue

        # Area — li with fa-arrows-alt icon
        area_m2 = None
        area_icon = card.find('i', class_='fa-arrows-alt')
        if area_icon:
            li = area_icon.find_parent('li') or area_icon.parent
            area_m2 = _parse_area(li.get_text(strip=True))
        if area_m2 is None:
            skipped += 1
            logger.debug('domiclick(%s): missing area at %s, skipping', base_url, source_url)
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
        logger.debug('domiclick(%s): skipped %d listings (non-USD or incomplete)', base_url, skipped)

    return results


def scrape(base_url: str, max_pages: int = 150, browser=None) -> list:
    """Scrape a Domiclick site and return all listing dicts.

    Args:
        base_url: Site base URL, e.g. 'https://www.apartamentosrd.com.do'.
        max_pages: Maximum pages to fetch before stopping.
        browser: Optional Playwright browser instance (for testing). If None,
                 a new Chromium instance is launched.

    Returns:
        Aggregated list of listing dicts from all pages.
    """
    from playwright.sync_api import sync_playwright

    def _fetch_page(browser_instance, page_num):
        url = LISTING_URL_TMPL.format(base=base_url, page=page_num)
        page = browser_instance.new_page()
        try:
            page.goto(url, wait_until='networkidle', timeout=30_000)
            return page.content()
        finally:
            page.close()

    def _run(browser_instance):
        results = []
        for page_num in range(1, max_pages + 1):
            html = _fetch_page(browser_instance, page_num)
            page_results = parse_listings(html, base_url)
            if not page_results:
                logger.debug('domiclick(%s): no cards on page %d, stopping', base_url, page_num)
                break
            results.extend(page_results)
        return results

    if browser is not None:
        return _run(browser)

    with sync_playwright() as p:
        browser_instance = p.chromium.launch(headless=True)
        try:
            return _run(browser_instance)
        finally:
            browser_instance.close()
