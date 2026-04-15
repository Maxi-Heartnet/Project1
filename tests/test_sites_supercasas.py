"""Tests for scraper/sites/supercasas.py — all HTTP is mocked, no real network calls."""
import types
from unittest.mock import patch

import pytest

from scraper.sites.supercasas import scrape, SUPERCASAS_BASE

# ---------------------------------------------------------------------------
# HTML fixtures
# ---------------------------------------------------------------------------

# Fixtures match the current supercasas.com DOM (as of 2026-04):
# - Cards are li.special inside #bigsearch-results-inner-results
# - .title1 = sector/location, .title2 = price ("Venta: US$ N")
# - Bedrooms shown as "Habitaciones : N", area as "N Mt2"
USD_CARD = """
<li class="special">
  <a href="/apartamentos-venta-piantini/1385955/">
    <div class="title1">Piantini</div>
    <div class="title2">Venta: US$ 140,000</div>
    <span>Apartamento en Piantini</span>
    <span>Habitaciones : 3</span>
    <span>Construcción : 95 Mt2</span>
  </a>
</li>
"""

RD_CARD = """
<li class="special">
  <a href="/apartamentos-venta-naco/999999/">
    <div class="title1">Naco</div>
    <div class="title2">Venta: RD$ 9,500,000</div>
    <span>Apartamento en Naco</span>
    <span>Habitaciones : 2</span>
    <span>Construcción : 80 Mt2</span>
  </a>
</li>
"""

NO_BEDROOMS_CARD = """
<li class="special">
  <a href="/apartamentos-venta-evaristo/111111/">
    <div class="title1">Evaristo Morales</div>
    <div class="title2">Venta: US$ 200,000</div>
    <span>Apartamento Evaristo Morales</span>
    <span>Construcción : 120 Mt2</span>
  </a>
</li>
"""

RESULTS_DIV = '<div id="bigsearch-results-inner-results"><ul>{cards}</ul></div>'
PAGE_WITH_CARDS = f'<html><body>{RESULTS_DIV.format(cards=USD_CARD)}</body></html>'
PAGE_EMPTY = f'<html><body>{RESULTS_DIV.format(cards="")}</body></html>'
PAGE_RD_ONLY = f'<html><body>{RESULTS_DIV.format(cards=RD_CARD)}</body></html>'
PAGE_NO_BEDROOMS = f'<html><body>{RESULTS_DIV.format(cards=NO_BEDROOMS_CARD)}</body></html>'
PAGE_TWO_CARDS = f'<html><body>{RESULTS_DIV.format(cards=USD_CARD + USD_CARD.replace("1385955", "2222222"))}</body></html>'


def _mock_response(html):
    """Create a minimal mock response object with .text and .content."""
    resp = types.SimpleNamespace()
    resp.text = html
    resp.content = html.encode('utf-8')
    resp.raise_for_status = lambda: None
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSupercasasScrape:

    def test_scrape_returns_list(self):
        """Patched page returns valid HTML; result is a list."""
        with patch('scraper.sites.supercasas.requests.get',
                   return_value=_mock_response(PAGE_WITH_CARDS)):
            result = scrape(max_pages=1)
        assert isinstance(result, list)

    def test_scrape_each_listing_has_required_keys(self):
        """All 6 required keys are present in every result dict."""
        with patch('scraper.sites.supercasas.requests.get',
                   return_value=_mock_response(PAGE_WITH_CARDS)):
            result = scrape(max_pages=1)
        assert len(result) > 0
        required_keys = {'price', 'sector', 'property_type', 'bedrooms', 'area_m2', 'source_url'}
        for listing in result:
            assert set(listing.keys()) == required_keys

    def test_scrape_source_url_is_absolute(self):
        """All source_url values start with https://."""
        with patch('scraper.sites.supercasas.requests.get',
                   return_value=_mock_response(PAGE_WITH_CARDS)):
            result = scrape(max_pages=1)
        assert len(result) > 0
        for listing in result:
            assert listing['source_url'].startswith('https://')

    def test_scrape_source_url_includes_base(self):
        """Relative hrefs are prepended with SUPERCASAS_BASE."""
        with patch('scraper.sites.supercasas.requests.get',
                   return_value=_mock_response(PAGE_WITH_CARDS)):
            result = scrape(max_pages=1)
        assert len(result) > 0
        for listing in result:
            assert listing['source_url'].startswith(SUPERCASAS_BASE)

    def test_scrape_skips_non_usd_listings(self):
        """Page with only RD$ listings returns an empty list."""
        with patch('scraper.sites.supercasas.requests.get',
                   return_value=_mock_response(PAGE_RD_ONLY)):
            result = scrape(max_pages=1)
        assert result == []

    def test_scrape_skips_listing_with_missing_field(self):
        """Card missing bedrooms is excluded from results."""
        with patch('scraper.sites.supercasas.requests.get',
                   return_value=_mock_response(PAGE_NO_BEDROOMS)):
            result = scrape(max_pages=1)
        assert result == []

    def test_scrape_empty_page_stops_pagination(self):
        """When page 2 returns no cards, scraping stops after page 1."""
        call_count = [0]

        def fake_get(url, **kwargs):
            call_count[0] += 1
            # First call (skip=0) returns cards; subsequent calls return empty
            if call_count[0] == 1:
                return _mock_response(PAGE_WITH_CARDS)
            return _mock_response(PAGE_EMPTY)

        with patch('scraper.sites.supercasas.requests.get', side_effect=fake_get):
            result = scrape(max_pages=5)

        # Only page 1 contributed listings
        assert len(result) == 1
        assert call_count[0] == 2  # fetched page 1 (cards) + page 2 (empty → stop)
