"""casaspb.com site parser.

Scrapes real estate listings from casaspb.com and returns a list of dicts
with 6 keys: price, sector, property_type, bedrooms, area_m2, source_url.
Non-USD listings and field-incomplete listings are skipped.

Selectors spiked on 2026-04-08. Both casaspb.com and miscasasrd.com run on
the EasyBroker platform; the structure is identical except for card selector
(div.thumbnail vs li.property-listing.clearfix) and location approach.
"""
import json
import logging
import re
import time

import requests
from bs4 import BeautifulSoup

from scraper.scraper import parse_price_usd

logger = logging.getLogger(__name__)

CASASPB_BASE = 'https://www.casaspb.com'
LISTING_URL_FIRST = 'https://www.casaspb.com/properties'
LISTING_URL_PAGE = 'https://www.casaspb.com/properties?page={page}&web_page=properties'

CARD_SELECTOR = 'div.thumbnail'
PRICE_SELECTOR = 'span.listing-type-price'

# Matches "Apartamento en " / "Casa en " / "Villa en " etc.
_PROPERTY_TYPE_PREFIX_RE = re.compile(r'^[A-Za-záéíóúÁÉÍÓÚüÜñÑ]+ en ', re.IGNORECASE)

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}


def _parse_bedrooms(text):
    """Return bedroom count from text containing bed/hab/dormitorio/recámara indicator, or None."""
    m = re.search(r'(\d+)\s*(?:hab|bed|dormitorio|rec[aá]mara)', text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _parse_area(text):
    """Return area in m2 from text containing m²/m2/mt2 indicator, or None."""
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


def _parse_sector(card) -> str | None:
    """Extract sector from a listing card element.

    Attempts data-popover-data JSON first (EasyBroker platform feature).
    Falls back to div.caption > span text: strips 'PropertyType en ' prefix,
    splits on ', ', and returns the first non-empty segment.
    """
    # Attempt 1: data-popover-data JSON attribute
    raw = card.get('data-popover-data')
    if raw:
        try:
            data = json.loads(raw)
            location = data.get('location') or data.get('sector')
            if location and location.strip():
                return location.strip()
        except (json.JSONDecodeError, AttributeError):
            pass

    # Attempt 2: div.caption > span text
    caption_span = card.select_one('div.caption > span')
    if caption_span:
        text = caption_span.get_text(separator=' ', strip=True)
        # Strip "Apartamento en " / "Casa en " / etc.
        text = _PROPERTY_TYPE_PREFIX_RE.sub('', text)
        parts = [p.strip() for p in text.split(',') if p.strip()]
        if parts:
            return parts[0]

    return None


def _fetch_page(page_num: int) -> str:
    """Fetch one page of casaspb.com listings and return HTML text."""
    if page_num == 1:
        url = LISTING_URL_FIRST
    else:
        url = LISTING_URL_PAGE.format(page=page_num)
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.text


def scrape(max_pages: int = 150) -> list:
    """Scrape casaspb.com and return a list of listing dicts.

    Each dict has 6 keys: price, sector, property_type, bedrooms, area_m2, source_url.
    Non-USD listings and field-incomplete listings are skipped.
    Stops early when a page returns no div.thumbnail cards.
    """
    results = []
    skipped = 0

    for page_num in range(1, max_pages + 1):
        html = _fetch_page(page_num)
        soup = BeautifulSoup(html, 'html.parser')
        cards = soup.select(CARD_SELECTOR)

        if not cards:
            logger.debug('casaspb: no cards on page %d, stopping', page_num)
            break

        for card in cards:
            # Price
            price_el = card.select_one(PRICE_SELECTOR)
            price_text = price_el.get_text(strip=True) if price_el else ''
            price = parse_price_usd(price_text)
            if price is None:
                skipped += 1
                logger.debug('casaspb: non-USD or missing price, skipping')
                continue

            # Source URL
            anchor = card.select_one('a.related-property[href]')
            if not anchor:
                anchor = card.find('a', href=True)
            if not anchor:
                skipped += 1
                logger.debug('casaspb: card has no anchor, skipping')
                continue
            href = anchor['href']
            source_url = href if href.startswith('http') else CASASPB_BASE + href

            # Full card text for detail extraction
            card_text = card.get_text(separator=' ', strip=True)

            # Sector
            sector = _parse_sector(card)
            if not sector:
                skipped += 1
                logger.debug('casaspb: missing sector at %s, skipping', source_url)
                continue

            # Property type from title/anchor text
            title_el = card.find('a', href=True)
            title_text = title_el.get_text(strip=True) if title_el else card_text
            property_type = _infer_property_type(title_text)

            # Bedrooms
            bedrooms = _parse_bedrooms(card_text)
            if bedrooms is None:
                skipped += 1
                logger.debug('casaspb: missing bedrooms at %s, skipping', source_url)
                continue

            # Area
            area_m2 = _parse_area(card_text)
            if area_m2 is None:
                skipped += 1
                logger.debug('casaspb: missing area at %s, skipping', source_url)
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
        logger.debug('casaspb: skipped %d listings (non-USD or incomplete)', skipped)

    return results
