"""Tests for scraper/sites/casaspb.py — all HTTP is mocked, no real network calls."""
import json
import types
from unittest.mock import patch

import pytest

from scraper.sites.casaspb import scrape, _parse_sector, CASASPB_BASE
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# HTML fixtures
# ---------------------------------------------------------------------------
# Matches casaspb.com DOM as spiked on 2026-04-08:
# - Cards are div.thumbnail
# - span.listing-type-price for price ("$420,000 USD" or "RD$ 56,480,000 DOP")
# - div.caption > span for location text ("Apartamento en Playa Cosón, Las Terrenas")
# - a.related-property[href*="/property/"] for anchor

USD_CARD = """
<div class="thumbnail">
  <a class="related-property" href="/property/12345/">
    <img src="/img.jpg" />
  </a>
  <div class="caption">
    <span>Apartamento en Playa Cosón, Las Terrenas</span>
    <span class="listing-type-price">$420,000 USD</span>
    <p>3 hab | 95 m²</p>
  </div>
</div>
"""

DOP_CARD = """
<div class="thumbnail">
  <a class="related-property" href="/property/99999/">
    <img src="/img.jpg" />
  </a>
  <div class="caption">
    <span>Casa en Santo Domingo, Distrito Nacional</span>
    <span class="listing-type-price">RD$ 56,480,000 DOP</span>
    <p>4 hab | 180 m²</p>
  </div>
</div>
"""

NO_BEDROOMS_CARD = """
<div class="thumbnail">
  <a class="related-property" href="/property/11111/">
    <img src="/img.jpg" />
  </a>
  <div class="caption">
    <span>Apartamento en Piantini, Santo Domingo</span>
    <span class="listing-type-price">$350,000 USD</span>
    <p>120 m²</p>
  </div>
</div>
"""

NO_AREA_CARD = """
<div class="thumbnail">
  <a class="related-property" href="/property/22222/">
    <img src="/img.jpg" />
  </a>
  <div class="caption">
    <span>Apartamento en Naco, Santo Domingo</span>
    <span class="listing-type-price">$280,000 USD</span>
    <p>2 hab</p>
  </div>
</div>
"""

DATA_POPOVER_CARD = (
    '<div class="thumbnail" data-popover-data=\'{"location": "Punta Cana", "price": 500000}\'>'
    '<a class="related-property" href="/property/33333/"><img src="/img.jpg" /></a>'
    '<div class="caption">'
    '<span>Villa en Bavaro, Punta Cana</span>'
    '<span class="listing-type-price">$500,000 USD</span>'
    '<p>3 hab | 200 m\u00b2</p>'
    '</div>'
    '</div>'
)

PAGE_WITH_CARD = '<html><body>{}</body></html>'.format(USD_CARD)
PAGE_EMPTY = '<html><body><div class="container"></div></body></html>'
PAGE_DOP_ONLY = '<html><body>{}</body></html>'.format(DOP_CARD)
PAGE_NO_BEDROOMS = '<html><body>{}</body></html>'.format(NO_BEDROOMS_CARD)
PAGE_NO_AREA = '<html><body>{}</body></html>'.format(NO_AREA_CARD)
PAGE_TWO_USD_ONE_DOP = '<html><body>{}</body></html>'.format(
    USD_CARD + USD_CARD.replace('12345', '54321') + DOP_CARD
)


def _mock_response(html):
    resp = types.SimpleNamespace()
    resp.text = html
    resp.raise_for_status = lambda: None
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCasaspbScrape:

    def test_returns_list(self):
        with patch('scraper.sites.casaspb.requests.get',
                   return_value=_mock_response(PAGE_WITH_CARD)):
            result = scrape(max_pages=1)
        assert isinstance(result, list)

    def test_each_listing_has_required_keys(self):
        with patch('scraper.sites.casaspb.requests.get',
                   return_value=_mock_response(PAGE_WITH_CARD)):
            result = scrape(max_pages=1)
        assert len(result) > 0
        required = {'price', 'sector', 'property_type', 'bedrooms', 'area_m2', 'source_url'}
        for listing in result:
            assert set(listing.keys()) == required

    def test_source_url_absolute_with_base(self):
        with patch('scraper.sites.casaspb.requests.get',
                   return_value=_mock_response(PAGE_WITH_CARD)):
            result = scrape(max_pages=1)
        assert len(result) > 0
        for listing in result:
            assert listing['source_url'].startswith(CASASPB_BASE)

    def test_skips_dop_listings(self):
        with patch('scraper.sites.casaspb.requests.get',
                   return_value=_mock_response(PAGE_DOP_ONLY)):
            result = scrape(max_pages=1)
        assert result == []

    def test_skips_missing_bedrooms(self):
        with patch('scraper.sites.casaspb.requests.get',
                   return_value=_mock_response(PAGE_NO_BEDROOMS)):
            result = scrape(max_pages=1)
        assert result == []

    def test_skips_missing_area(self):
        with patch('scraper.sites.casaspb.requests.get',
                   return_value=_mock_response(PAGE_NO_AREA)):
            result = scrape(max_pages=1)
        assert result == []

    def test_empty_page_stops_pagination(self):
        call_count = [0]

        def fake_get(url, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _mock_response(PAGE_WITH_CARD)
            return _mock_response(PAGE_EMPTY)

        with patch('scraper.sites.casaspb.requests.get', side_effect=fake_get):
            result = scrape(max_pages=5)

        assert len(result) == 1
        assert call_count[0] == 2

    def test_multi_card_filters_dop(self):
        """Two USD + one DOP card → only 2 results."""
        with patch('scraper.sites.casaspb.requests.get',
                   return_value=_mock_response(PAGE_TWO_USD_ONE_DOP)):
            result = scrape(max_pages=1)
        assert len(result) == 2


class TestCasaspbSectorParsing:

    def test_caption_span_strips_property_type_prefix(self):
        """'Apartamento en Playa Cosón, Las Terrenas' → first segment 'Playa Cosón'."""
        soup = BeautifulSoup(USD_CARD, 'html.parser')
        card = soup.select_one('div.thumbnail')
        sector = _parse_sector(card)
        assert sector == 'Playa Cosón'

    def test_data_popover_data_takes_priority(self):
        """data-popover-data JSON is used when present."""
        soup = BeautifulSoup(DATA_POPOVER_CARD, 'html.parser')
        card = soup.select_one('div.thumbnail')
        sector = _parse_sector(card)
        assert sector == 'Punta Cana'

    def test_caption_without_comma_returns_full_text(self):
        """Single-segment location (no comma) returns the full stripped text."""
        html = """
        <div class="thumbnail">
          <div class="caption"><span>Piantini</span></div>
        </div>
        """
        soup = BeautifulSoup(html, 'html.parser')
        card = soup.select_one('div.thumbnail')
        sector = _parse_sector(card)
        assert sector == 'Piantini'

    def test_no_location_returns_none(self):
        """Card with no caption span and no data-popover-data → None."""
        html = '<div class="thumbnail"><a href="/property/1/"></a></div>'
        soup = BeautifulSoup(html, 'html.parser')
        card = soup.select_one('div.thumbnail')
        sector = _parse_sector(card)
        assert sector is None
