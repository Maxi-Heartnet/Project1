from unittest.mock import patch
from bs4 import BeautifulSoup

from scraper.sites.corotos import scrape, COROTOS_BASE, CARD_SELECTOR

# Minimal HTML for a single USD listing card matching corotos.com.do DOM
USD_CARD_HTML = """
<div class="listing-item col s12 l4">
  <a href="/anuncio/apartamento-en-piantini-abc123">
    <h3>Apartamento en Venta en Piantini</h3>
  </a>
  <a class="item__price" href="/anuncio/apartamento-en-piantini-abc123">
    <div class="card-row">
      <span class="item__currency">US$</span>
      <span class="item__price-amount">185,000</span>
    </div>
  </a>
  <a class="info-wrapper" href="/anuncio/apartamento-en-piantini-abc123">
    <div class="location-more">
      <span class="listing-text-gray card-right-text">Piantini</span>
    </div>
    <div class="real-estate-more">
      <span>3 hab</span>
      <span>120 m²</span>
    </div>
  </a>
</div>
"""

RD_CARD_HTML = """
<div class="listing-item col s12 l4">
  <a href="/anuncio/apartamento-rd-123">
    <h3>Apartamento en Naco</h3>
  </a>
  <a class="item__price" href="/anuncio/apartamento-rd-123">
    <div class="card-row">
      <span class="item__currency">RD$</span>
      <span class="item__price-amount">8,000,000</span>
    </div>
  </a>
  <a class="info-wrapper" href="/anuncio/apartamento-rd-123">
    <div class="location-more">
      <span class="listing-text-gray card-right-text">Naco</span>
    </div>
    <div class="real-estate-more">
      <span>2 hab</span>
      <span>90 m²</span>
    </div>
  </a>
</div>
"""

NO_PRICE_CARD_HTML = """
<div class="listing-item col s12 l4">
  <a href="/anuncio/casa-sin-precio-xyz">
    <h3>Casa en Bella Vista</h3>
  </a>
  <a class="info-wrapper" href="/anuncio/casa-sin-precio-xyz">
    <div class="location-more">
      <span class="listing-text-gray card-right-text">Bella Vista</span>
    </div>
  </a>
</div>
"""

PAGE_WITH_CARDS = f'<html><body>{USD_CARD_HTML}</body></html>'
PAGE_EMPTY = '<html><body><p>no listings</p></body></html>'
PAGE_TWO_CARDS = f'<html><body>{USD_CARD_HTML}{USD_CARD_HTML.replace("abc123", "def456")}</body></html>'


def test_scrape_returns_list():
    with patch('scraper.sites.corotos.fetch_page', return_value=PAGE_WITH_CARDS):
        result = scrape(max_pages=1)
    assert isinstance(result, list)


def test_scrape_each_listing_has_required_keys():
    with patch('scraper.sites.corotos.fetch_page', return_value=PAGE_WITH_CARDS):
        result = scrape(max_pages=1)
    assert len(result) > 0
    for listing in result:
        assert set(listing.keys()) == {'price', 'sector', 'property_type', 'bedrooms', 'area_m2', 'source_url'}


def test_scrape_source_url_is_absolute():
    with patch('scraper.sites.corotos.fetch_page', return_value=PAGE_WITH_CARDS):
        result = scrape(max_pages=1)
    assert len(result) > 0
    for listing in result:
        assert listing['source_url'].startswith('https://')


def test_scrape_source_url_includes_base():
    with patch('scraper.sites.corotos.fetch_page', return_value=PAGE_WITH_CARDS):
        result = scrape(max_pages=1)
    assert result[0]['source_url'].startswith(COROTOS_BASE)


def test_scrape_skips_non_usd_listings():
    page = f'<html><body>{RD_CARD_HTML}</body></html>'
    with patch('scraper.sites.corotos.fetch_page', return_value=page):
        result = scrape(max_pages=1)
    assert result == []


def test_scrape_skips_listing_with_missing_field():
    page = f'<html><body>{NO_PRICE_CARD_HTML}</body></html>'
    with patch('scraper.sites.corotos.fetch_page', return_value=page):
        result = scrape(max_pages=1)
    assert result == []


def test_scrape_empty_page_stops_pagination():
    pages = {1: PAGE_WITH_CARDS, 2: PAGE_EMPTY}
    with patch('scraper.sites.corotos.fetch_page', side_effect=lambda p, **kw: pages.get(p, PAGE_EMPTY)):
        result = scrape(max_pages=5)
    # Only page 1 contributed listings
    assert len(result) == 1


def test_scrape_real_cache_returns_list():
    """Integration test using real cached HTML from data/raw/."""
    result = scrape(max_pages=1, use_cache=True)
    assert isinstance(result, list)
    assert len(result) > 0


def test_scrape_real_cache_all_have_source_url():
    result = scrape(max_pages=1, use_cache=True)
    for listing in result:
        assert listing['source_url'].startswith('https://')


def test_scrape_real_cache_all_have_required_keys():
    result = scrape(max_pages=1, use_cache=True)
    for listing in result:
        assert set(listing.keys()) == {'price', 'sector', 'property_type', 'bedrooms', 'area_m2', 'source_url'}
