"""Corotos.com.do site parser.

Imports field-parsing helpers from the reference implementation (scraper.scraper)
and reimplements the pagination loop to add source_url to each listing dict.
"""
import logging
import time

from bs4 import BeautifulSoup

from scraper.scraper import fetch_page, parse_listing

logger = logging.getLogger(__name__)

# Selector for the listing cards on corotos.com.do
CARD_SELECTOR = 'div.listing-item'

# Base URL for constructing canonical absolute source_url values
COROTOS_BASE = 'https://www.corotos.com.do'


def scrape(max_pages: int = 30, use_cache: bool = True) -> list:
    """Scrape corotos.com.do and return a list of listing dicts.

    Each dict has 6 keys: price, sector, property_type, bedrooms, area_m2, source_url.
    Non-USD listings (price=None from parse_listing) and field-incomplete listings
    are skipped. Stops early if a page returns no cards.

    Returns a list of dicts ready for insert_listings().
    """
    results = []
    skipped = 0

    for page_num in range(1, max_pages + 1):
        html = fetch_page(page_num, use_cache=use_cache)
        soup = BeautifulSoup(html, 'html.parser')
        cards = soup.select(CARD_SELECTOR)

        if not cards:
            logger.debug('corotos: no cards on page %d, stopping', page_num)
            break

        for card in cards:
            # Extract the listing URL from the first anchor in the card
            anchor = card.find('a', href=True)
            if not anchor:
                skipped += 1
                logger.debug('corotos: card has no anchor, skipping')
                continue

            href = anchor['href']
            if href.startswith('http'):
                source_url = href
            else:
                source_url = COROTOS_BASE + href

            # parse_listing returns None price for non-USD listings
            parsed = parse_listing(card)
            if any(v is None for v in parsed.values()):
                skipped += 1
                logger.debug('corotos: incomplete listing at %s, skipping', source_url)
                continue

            results.append({
                'price': parsed['price'],
                'sector': parsed['sector'],
                'property_type': parsed['property_type'],
                'bedrooms': parsed['bedrooms'],
                'area_m2': parsed['area_m2'],
                'source_url': source_url,
            })

        if not use_cache:
            time.sleep(1)

    if skipped:
        logger.debug('corotos: skipped %d listings (non-USD or incomplete)', skipped)

    return results
