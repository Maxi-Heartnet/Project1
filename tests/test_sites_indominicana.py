"""Tests for scraper/sites/indominicana.py — all tests use mocked requests.get."""
from unittest.mock import patch, MagicMock

from scraper.sites.indominicana import scrape, INDOMINICANA_BASE, CARD_SELECTOR

# ---------------------------------------------------------------------------
# HTML fixtures
# ---------------------------------------------------------------------------

USD_CARD_HTML = """
<div class="property-container">
  <a href="/propiedades/apartamento-en-piantini-123">
    <h2>Apartamento en Venta</h2>
  </a>
  <a href="/barrios/piantini">Evaristo Morales, Santo Domingo</a>
  <div class="property-price">US$ 185,000</div>
  <div class="property-details">3 hab | 120 m²</div>
</div>
"""

RD_CARD_HTML = """
<div class="property-container">
  <a href="/propiedades/apartamento-naco-456">
    <h2>Apartamento en Venta</h2>
  </a>
  <a href="/barrios/naco">Naco, Santo Domingo</a>
  <div class="property-price">RD$ 8,000,000</div>
  <div class="property-details">2 hab | 90 m²</div>
</div>
"""

# Missing bedrooms — incomplete listing
INCOMPLETE_CARD_HTML = """
<div class="property-container">
  <a href="/propiedades/casa-bella-vista-789">
    <h2>Casa en Venta</h2>
  </a>
  <a href="/barrios/bella-vista">Bella Vista, Santo Domingo</a>
  <div class="property-price">US$ 250,000</div>
  <div class="property-details">200 m²</div>
</div>
"""

PAGE_WITH_CARD = f'<html><body>{USD_CARD_HTML}</body></html>'
PAGE_WITH_RD_CARD = f'<html><body>{RD_CARD_HTML}</body></html>'
PAGE_WITH_INCOMPLETE_CARD = f'<html><body>{INCOMPLETE_CARD_HTML}</body></html>'
PAGE_EMPTY = '<html><body><p>No se encontraron propiedades.</p></body></html>'


def _make_response(html: str) -> MagicMock:
    """Return a mock requests.Response with the given HTML text."""
    mock_resp = MagicMock()
    mock_resp.text = html
    mock_resp.raise_for_status.return_value = None
    return mock_resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_scrape_returns_list():
    """scrape() always returns a list."""
    with patch('scraper.sites.indominicana.requests.get',
               return_value=_make_response(PAGE_WITH_CARD)):
        result = scrape(max_pages=1)
    assert isinstance(result, list)


def test_scrape_each_listing_has_required_keys():
    """Every returned listing dict contains exactly the 6 required keys."""
    with patch('scraper.sites.indominicana.requests.get',
               return_value=_make_response(PAGE_WITH_CARD)):
        result = scrape(max_pages=1)
    assert len(result) > 0
    required = {'price', 'sector', 'property_type', 'bedrooms', 'area_m2', 'source_url'}
    for listing in result:
        assert set(listing.keys()) == required


def test_scrape_source_url_is_absolute():
    """source_url in every listing must be an absolute https:// URL."""
    with patch('scraper.sites.indominicana.requests.get',
               return_value=_make_response(PAGE_WITH_CARD)):
        result = scrape(max_pages=1)
    assert len(result) > 0
    for listing in result:
        assert listing['source_url'].startswith('https://')


def test_scrape_skips_non_usd_listings():
    """Listings priced in RD$ are silently skipped."""
    with patch('scraper.sites.indominicana.requests.get',
               return_value=_make_response(PAGE_WITH_RD_CARD)):
        result = scrape(max_pages=1)
    assert result == []


def test_scrape_skips_listing_with_missing_field():
    """Listings that lack a required field (e.g. bedrooms) are skipped."""
    with patch('scraper.sites.indominicana.requests.get',
               return_value=_make_response(PAGE_WITH_INCOMPLETE_CARD)):
        result = scrape(max_pages=1)
    assert result == []


def test_scrape_empty_page_stops_pagination():
    """Pagination halts as soon as a page returns no property-container cards."""
    call_count = 0

    def side_effect(url, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_response(PAGE_WITH_CARD)
        return _make_response(PAGE_EMPTY)

    with patch('scraper.sites.indominicana.requests.get', side_effect=side_effect):
        result = scrape(max_pages=5, use_cache=True)

    # Only the first page contributed a listing; iteration stopped at page 2
    assert len(result) == 1
    assert call_count == 2
