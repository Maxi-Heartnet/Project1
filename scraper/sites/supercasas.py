"""supercasas.com site parser.

Scrapes real estate listings from supercasas.com and returns a list of dicts
with 6 keys: price, sector, property_type, bedrooms, area_m2, source_url.
Non-USD listings and field-incomplete listings are skipped.
"""
import logging
import re
import time

import requests
from bs4 import BeautifulSoup

from scraper.scraper import parse_price_usd

logger = logging.getLogger(__name__)

SUPERCASAS_BASE = 'https://www.supercasas.com'
LISTING_URL = 'https://www.supercasas.com/buscar/?Tipo=2&PagingPageSkip={skip}'

CARD_SELECTOR = '#bigsearch-results-inner-results li.special'
PRICE_SELECTOR = '.title2'
LOCATION_SELECTOR = '.title1'

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}


def _first_number(text):
    """Return the first integer found in text, or None."""
    m = re.search(r'\d+(?:[.,]\d+)*', text)
    if not m:
        return None
    return int(re.sub(r'[.,]', '', m.group()))


def _parse_bedrooms(text):
    """Return bedroom count from 'Habitaciones : N' format, or None."""
    # New DOM format: "Habitaciones : 3"
    m = re.search(r'Habitaciones\s*:\s*(\d+)', text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # Fallback: "N hab" format
    m = re.search(r'(\d+)\s*hab', text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _parse_area(text):
    """Return area in m2 from 'Construccion : N Mt2' or standard 'N m²', or None."""
    # Supercasas uses 'Mt2' (e.g. 'Construcción : 239 Mt2')
    m = re.search(r'(\d+(?:[.,]\d+)?)\s*[Mm]t?[2²]', text)
    if m:
        return float(re.sub(r',', '.', m.group(1)))
    return None


def _infer_property_type(text):
    """Infer property type from title text; default to 'apartment'."""
    lower = text.lower()
    if 'casa' in lower or 'house' in lower or 'villa' in lower:
        return 'house'
    return 'apartment'


def _fetch_page(skip: int) -> str:
    """Fetch one page of supercasas.com listings and return HTML text."""
    url = LISTING_URL.format(skip=skip)
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.text


def scrape(max_pages: int = 50) -> list:
    """Scrape supercasas.com and return a list of listing dicts.

    Each dict has 6 keys: price, sector, property_type, bedrooms, area_m2, source_url.
    Non-USD listings and field-incomplete listings are skipped.
    Stops early when a page returns no li.normal cards.
    """
    results = []
    skipped = 0

    for page_num in range(max_pages):
        skip = page_num  # PagingPageSkip is zero-based page index
        html = _fetch_page(skip)
        soup = BeautifulSoup(html, 'html.parser')
        cards = soup.select(CARD_SELECTOR)

        if not cards:
            logger.debug('supercasas: no cards on skip=%d, stopping', skip)
            break

        for card in cards:
            # Price
            price_el = card.select_one(PRICE_SELECTOR)
            price_text = price_el.get_text(strip=True) if price_el else ''
            price = parse_price_usd(price_text)
            if price is None:
                skipped += 1
                logger.debug('supercasas: non-USD or missing price, skipping')
                continue

            # Sector
            loc_el = card.select_one(LOCATION_SELECTOR)
            sector = loc_el.get_text(strip=True) if loc_el else None
            if not sector:
                skipped += 1
                logger.debug('supercasas: missing sector, skipping')
                continue

            # Source URL
            anchor = card.select_one('a[href]')
            if not anchor:
                skipped += 1
                logger.debug('supercasas: card has no anchor, skipping')
                continue
            href = anchor['href']
            source_url = href if href.startswith('http') else SUPERCASAS_BASE + href

            # Full card text for detail extraction
            card_text = card.get_text(separator=' ', strip=True)

            # Property type from title or card text
            title_el = card.select_one('a')
            title_text = title_el.get_text(strip=True) if title_el else card_text
            property_type = _infer_property_type(title_text)

            # Bedrooms
            bedrooms = _parse_bedrooms(card_text)
            if bedrooms is None:
                skipped += 1
                logger.debug('supercasas: missing bedrooms at %s, skipping', source_url)
                continue

            # Area
            area_m2 = _parse_area(card_text)
            if area_m2 is None:
                skipped += 1
                logger.debug('supercasas: missing area at %s, skipping', source_url)
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
        logger.debug('supercasas: skipped %d listings (non-USD or incomplete)', skipped)

    return results
