"""indominicana.com site parser.

Scrapes real estate listings from indominicana.com and returns a list of dicts
with 6 keys: price, sector, property_type, bedrooms, area_m2, source_url.
Non-USD listings and field-incomplete listings are skipped.
"""
import logging
import re
import time

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

INDOMINICANA_BASE = 'https://indominicana.com'
FIRST_PAGE_URL = 'https://indominicana.com/propiedades/venta/apartamentos'
PAGE_URL = 'https://indominicana.com/propiedades.php?status=sale&type[0]=apartamentos&page_no={page_no}'

CARD_SELECTOR = 'div.property-container'

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}


def _fetch_page(page_num: int) -> str:
    """Fetch one page of indominicana.com listings and return HTML text."""
    if page_num == 1:
        url = FIRST_PAGE_URL
    else:
        url = PAGE_URL.format(page_no=page_num)
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.text


def _parse_price_usd(card_text):
    """Return integer USD price from card text, or None if not a USD listing."""
    m = re.search(r'US\$\s*([\d,]+)', card_text)
    if not m:
        return None
    return int(m.group(1).replace(',', ''))


def _parse_bedrooms(card_text):
    """Return bedroom count from card text, or None."""
    m = re.search(r'(\d+)\s*(?:hab(?:itacion(?:es)?)?|dormitorio(?:s)?|cuarto(?:s)?)',
                  card_text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _parse_area(card_text):
    """Return area in m2 from card text, or None."""
    m = re.search(r'(\d+(?:[.,]\d+)?)\s*m[²2]', card_text, re.IGNORECASE)
    if m:
        return float(m.group(1).replace(',', '.'))
    return None


def _infer_property_type(card_text):
    """Infer property type from card text; default to 'apartment'."""
    lower = card_text.lower()
    if 'casa' in lower or 'villa' in lower:
        return 'house'
    return 'apartment'


def scrape(max_pages: int = 50) -> list:
    """Scrape indominicana.com and return a list of listing dicts.

    Each dict has 6 keys: price, sector, property_type, bedrooms, area_m2, source_url.
    Non-USD listings and field-incomplete listings are skipped.
    Stops early when a page returns no div.property-container cards.
    """
    results = []
    skipped = 0

    for page_num in range(1, max_pages + 1):
        html = _fetch_page(page_num)
        soup = BeautifulSoup(html, 'html.parser')
        cards = soup.select(CARD_SELECTOR)

        if not cards:
            logger.debug('indominicana: no cards on page %d, stopping', page_num)
            break

        for card in cards:
            # Use no separator so m<sup>2</sup> stays as "m2" not "m 2"
            card_text = card.get_text(strip=True)

            # Price — skip non-USD listings
            price = _parse_price_usd(card_text)
            if price is None:
                skipped += 1
                logger.debug('indominicana: non-USD or missing price, skipping')
                continue

            # Sector — from location anchor (may start with comma, e.g. ", Santo Domingo Este")
            sector = None
            for a in card.find_all('a', href=True):
                a_text = a.get_text(strip=True)
                if not a_text:
                    continue
                # Strip leading/trailing commas then take first segment
                parts = [p.strip() for p in a_text.split(',') if p.strip()]
                if parts:
                    sector = parts[0]
                    break
            if not sector:
                skipped += 1
                logger.debug('indominicana: missing sector, skipping')
                continue

            # Source URL — first anchor with /propiedades/ in href
            anchor = card.select_one('a[href*="/propiedades/"]')
            if not anchor:
                skipped += 1
                logger.debug('indominicana: card has no property anchor, skipping')
                continue
            href = anchor['href']
            source_url = href if href.startswith('http') else INDOMINICANA_BASE + href

            # Property type from card title/text
            property_type = _infer_property_type(card_text)

            # Bedrooms
            bedrooms = _parse_bedrooms(card_text)
            if bedrooms is None:
                skipped += 1
                logger.debug('indominicana: missing bedrooms at %s, skipping', source_url)
                continue

            # Area
            area_m2 = _parse_area(card_text)
            if area_m2 is None:
                skipped += 1
                logger.debug('indominicana: missing area at %s, skipping', source_url)
                continue

            results.append({
                'price': price,
                'sector': sector,
                'property_type': property_type,
                'bedrooms': bedrooms,
                'area_m2': area_m2,
                'source_url': source_url,
            })

        time.sleep(1)

    if skipped:
        logger.debug('indominicana: skipped %d listings (non-USD or incomplete)', skipped)

    return results
