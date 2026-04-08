"""Tests for scraper/sites/miscasasrd.py — all HTTP is mocked, no real network calls."""
import types
from unittest.mock import patch

import pytest

from scraper.sites.miscasasrd import scrape, MISCASASRD_BASE

# ---------------------------------------------------------------------------
# HTML fixtures
# ---------------------------------------------------------------------------

USD_CARD = """
<li class="property-listing clearfix">
  <a class="related-property" href="/property/apartamento-piantini-420000">
    Apartamento en Piantini
  </a>
  <span class="listing-type-price">$420,000 USD</span>
  <div class="description">Piantini</div>
  <span>3 hab</span>
  <span>110 m²</span>
</li>
"""

RD_CARD = """
<li class="property-listing clearfix">
  <a class="related-property" href="/property/apartamento-naco-rd">
    Apartamento en Naco
  </a>
  <span class="listing-type-price">RD$ 5,900,000 DOP</span>
  <div class="description">Naco</div>
  <span>2 hab</span>
  <span>85 m²</span>
</li>
"""

NO_BEDROOMS_CARD = """
<li class="property-listing clearfix">
  <a class="related-property" href="/property/casa-bella-vista-nobed">
    Casa en Bella Vista
  </a>
  <span class="listing-type-price">$350,000 USD</span>
  <div class="description">Bella Vista</div>
  <span>200 m²</span>
</li>
"""

PAGE_WITH_CARDS = f'<html><body><ul>{USD_CARD}</ul></body></html>'
PAGE_EMPTY = '<html><body><ul></ul></body></html>'
PAGE_RD_ONLY = f'<html><body><ul>{RD_CARD}</ul></body></html>'
PAGE_NO_BEDROOMS = f'<html><body><ul>{NO_BEDROOMS_CARD}</ul></body></html>'


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

class TestMiscasasrdScrape:

    def test_scrape_returns_list(self):
        """Patched page returns valid HTML; result is a list."""
        with patch('scraper.sites.miscasasrd.requests.get',
                   return_value=_mock_response(PAGE_WITH_CARDS)):
            result = scrape(max_pages=1)
        assert isinstance(result, list)

    def test_scrape_each_listing_has_required_keys(self):
        """All 6 required keys are present in every result dict."""
        with patch('scraper.sites.miscasasrd.requests.get',
                   return_value=_mock_response(PAGE_WITH_CARDS)):
            result = scrape(max_pages=1)
        assert len(result) > 0
        required_keys = {'price', 'sector', 'property_type', 'bedrooms', 'area_m2', 'source_url'}
        for listing in result:
            assert set(listing.keys()) == required_keys

    def test_scrape_source_url_is_absolute(self):
        """All source_url values start with https://."""
        with patch('scraper.sites.miscasasrd.requests.get',
                   return_value=_mock_response(PAGE_WITH_CARDS)):
            result = scrape(max_pages=1)
        assert len(result) > 0
        for listing in result:
            assert listing['source_url'].startswith('https://')

    def test_scrape_source_url_includes_base(self):
        """Relative hrefs are prepended with MISCASASRD_BASE."""
        with patch('scraper.sites.miscasasrd.requests.get',
                   return_value=_mock_response(PAGE_WITH_CARDS)):
            result = scrape(max_pages=1)
        assert len(result) > 0
        for listing in result:
            assert listing['source_url'].startswith(MISCASASRD_BASE)

    def test_scrape_skips_non_usd_listings(self):
        """Page with only RD$ listings returns an empty list."""
        with patch('scraper.sites.miscasasrd.requests.get',
                   return_value=_mock_response(PAGE_RD_ONLY)):
            result = scrape(max_pages=1)
        assert result == []

    def test_scrape_skips_listing_with_missing_field(self):
        """Card missing bedrooms is excluded from results."""
        with patch('scraper.sites.miscasasrd.requests.get',
                   return_value=_mock_response(PAGE_NO_BEDROOMS)):
            result = scrape(max_pages=1)
        assert result == []

    def test_scrape_empty_page_stops_pagination(self):
        """When page 2 returns no cards, scraping stops after page 1."""
        call_count = [0]

        def fake_get(url, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _mock_response(PAGE_WITH_CARDS)
            return _mock_response(PAGE_EMPTY)

        with patch('scraper.sites.miscasasrd.requests.get', side_effect=fake_get):
            result = scrape(max_pages=5)

        # Only page 1 contributed listings
        assert len(result) == 1
        assert call_count[0] == 2  # fetched page 1 (cards) + page 2 (empty → stop)
