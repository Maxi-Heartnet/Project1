"""plusval.com.do site parser.

Scrapes real estate listings from Plusval Dominican Republic using
requests + BeautifulSoup. All listings on this site are USD-priced;
any RD$ listings encountered are skipped.
"""
import logging
import re
import time

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = 'https://plusval.com.do'
LISTING_URL = 'https://plusval.com.do/propiedades/venta'
PAGINATION_URL = 'https://plusval.com.do/propiedades/venta?page={n}'
CARD_SELECTOR = 'li.featured-property'

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    )
}


def _page_url(page_num: int) -> str:
    """Return the URL for the given 1-based page number."""
    if page_num == 1:
        return LISTING_URL
    return PAGINATION_URL.format(n=page_num)


def _parse_price(card) -> int | None:
    """Return USD price as int, or None if non-USD or unparseable."""
    price_tag = card.select_one('p.lead.font-pulpBold.text-primary-100')
    if not price_tag:
        return None

    price_text = price_tag.get_text(strip=True)

    # Skip peso listings
    if 'RD$' in price_text or 'DOP' in price_text:
        return None

    # Accept US$, $, or bare numeric after stripping currency symbols
    # e.g. "US$140,000" → 140000
    digits_only = re.sub(r'[^\d]', '', price_text)
    if not digits_only:
        return None

    try:
        return int(digits_only)
    except ValueError:
        return None


def _parse_sector(card) -> str | None:
    """Return sector name or None if not determinable."""
    # Preferred: short sector label
    label_tag = card.select_one('span.label.medium-label')
    if label_tag:
        sector = label_tag.get_text(strip=True)
        if sector:
            return sector

    # Fallback: extract from title (text after last comma)
    title_tag = card.select_one('h4.lead.font-pulpRegular')
    if title_tag:
        title_text = title_tag.get_text(strip=True)
        if ',' in title_text:
            return title_text.rsplit(',', 1)[-1].strip() or None

    return None


def _parse_property_type(card) -> str:
    """Infer property type from card title text."""
    title_tag = card.select_one('h4.lead.font-pulpRegular')
    title = title_tag.get_text(strip=True).lower() if title_tag else ''
    if 'apartamento' in title or 'apto' in title:
        return 'apartment'
    if 'casa' in title or 'villa' in title:
        return 'house'
    return 'apartment'


def _parse_bedrooms(card) -> int | None:
    """Return bedroom count or None if not found."""
    card_text = card.get_text()
    match = re.search(r'(\d+)\s*(?:hab|bed)', card_text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _parse_area(card) -> float | None:
    """Return area in m² or None if not found."""
    card_text = card.get_text()
    match = re.search(r'(\d[\d.,]*)\s*m[²2]', card_text, re.IGNORECASE)
    if match:
        raw = match.group(1).replace(',', '.')
        try:
            return float(raw)
        except ValueError:
            return None
    return None


def _parse_source_url(card) -> str | None:
    """Return absolute listing URL or None."""
    anchor = card.select_one('a.property[href*="/propiedad/"]')
    if not anchor:
        return None
    href = anchor['href']
    if href.startswith('https://'):
        return href
    if href.startswith('/'):
        return BASE_URL + href
    return BASE_URL + '/' + href


def scrape(max_pages: int = 50) -> list:
    """Scrape plusval.com.do and return a list of listing dicts.

    Each dict has 6 keys: price, sector, property_type, bedrooms, area_m2,
    source_url. Non-USD and field-incomplete listings are skipped.
    Stops early when a page returns no cards.
    """
    results = []
    skipped = 0

    for page_num in range(1, max_pages + 1):
        url = _page_url(page_num)
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            html = response.text
        except requests.RequestException as exc:
            logger.warning('plusval: request failed for page %d: %s', page_num, exc)
            break

        soup = BeautifulSoup(html, 'html.parser')
        cards = soup.select(CARD_SELECTOR)

        if not cards:
            logger.debug('plusval: no cards on page %d, stopping', page_num)
            break

        for card in cards:
            price = _parse_price(card)
            if price is None:
                skipped += 1
                logger.debug('plusval: skipping non-USD or unparseable price')
                continue

            sector = _parse_sector(card)
            property_type = _parse_property_type(card)
            bedrooms = _parse_bedrooms(card)
            area_m2 = _parse_area(card)
            source_url = _parse_source_url(card)

            if any(v is None for v in (sector, bedrooms, area_m2, source_url)):
                skipped += 1
                logger.debug('plusval: incomplete listing at %s, skipping', source_url)
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
        logger.debug('plusval: skipped %d listings (non-USD or incomplete)', skipped)

    return results
