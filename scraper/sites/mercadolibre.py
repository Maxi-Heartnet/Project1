"""inmuebles.mercadolibre.com.do site parser.

Scrapes real estate listings from MercadoLibre Dominican Republic using
requests + BeautifulSoup. Only USD listings are returned; peso (RD$/DOP)
listings are silently skipped.
"""
import logging
import re
import time

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = 'https://inmuebles.mercadolibre.com.do/'
PAGINATION_URL = 'https://inmuebles.mercadolibre.com.do/_Desde_{n}_NoIndex_True'
CARD_SELECTOR = 'div.ui-search-result__wrapper'

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
        return BASE_URL
    # Page 2 starts at offset 49, page 3 at 97, etc.
    offset = 1 + (page_num - 1) * 48
    return PAGINATION_URL.format(n=offset)


def _parse_price(card) -> int | None:
    """Return USD price as int, or None if non-USD or unparseable."""
    currency_tag = card.select_one('span.andes-money-amount__currency-symbol')
    amount_tag = card.select_one('span.andes-money-amount__fraction')

    if not currency_tag or not amount_tag:
        return None

    currency_text = currency_tag.get_text(strip=True)
    # Check the broader card text for RD$ or DOP signals
    card_text = card.get_text()
    if 'RD$' in card_text or 'DOP' in card_text:
        return None
    # Accept '$' as USD on this subdomain
    if currency_text not in ('$', 'USD'):
        return None

    amount_str = amount_tag.get_text(strip=True)
    # Remove thousands separators (commas or dots used as grouping)
    amount_str = re.sub(r'[.,]', '', amount_str)
    try:
        return int(amount_str)
    except ValueError:
        return None


def _parse_sector(card) -> str | None:
    """Return sector name (text before first comma) or None."""
    location_tag = card.select_one('span.poly-component__location')
    if not location_tag:
        return None
    text = location_tag.get_text(strip=True)
    return text.split(',')[0].strip() or None


def _parse_property_type(card) -> str:
    """Infer property type from listing title text."""
    title_tag = card.select_one('a.poly-component__title')
    title = title_tag.get_text(strip=True).lower() if title_tag else ''
    if 'apartamento' in title or 'apto' in title:
        return 'apartment'
    if 'casa' in title or 'villa' in title or 'townhouse' in title:
        return 'house'
    return 'apartment'


def _parse_bedrooms(card) -> int | None:
    """Return bedroom count or None if not found."""
    card_text = card.get_text()
    match = re.search(r'(\d+)\s*(?:hab|dorm|bed)', card_text, re.IGNORECASE)
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
    anchor = card.select_one('a.poly-component__title[href]')
    if not anchor:
        return None
    href = anchor['href']
    return href if href.startswith('https://') else None


def scrape(max_pages: int = 20) -> list:
    """Scrape inmuebles.mercadolibre.com.do and return a list of listing dicts.

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
            logger.warning('mercadolibre: request failed for page %d: %s', page_num, exc)
            break

        soup = BeautifulSoup(html, 'html.parser')
        cards = soup.select(CARD_SELECTOR)

        if not cards:
            logger.debug('mercadolibre: no cards on page %d, stopping', page_num)
            break

        for card in cards:
            price = _parse_price(card)
            if price is None:
                skipped += 1
                logger.debug('mercadolibre: skipping non-USD or unparseable price')
                continue

            sector = _parse_sector(card)
            property_type = _parse_property_type(card)
            bedrooms = _parse_bedrooms(card)
            area_m2 = _parse_area(card)
            source_url = _parse_source_url(card)

            if any(v is None for v in (sector, bedrooms, area_m2, source_url)):
                skipped += 1
                logger.debug('mercadolibre: incomplete listing at %s, skipping', source_url)
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
        logger.debug('mercadolibre: skipped %d listings (non-USD or incomplete)', skipped)

    return results
