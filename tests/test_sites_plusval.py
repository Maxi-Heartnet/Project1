"""Tests for scraper/sites/plusval.py.

All tests mock requests.get so no real network calls are made.
"""
from unittest.mock import patch, MagicMock

from scraper.sites.plusval import scrape

# ---------------------------------------------------------------------------
# HTML fixtures
# ---------------------------------------------------------------------------

USD_CARD_HTML = """
<li class="featured-property">
  <a class="property" href="/propiedad/apartamento-bella-vista-123">
    <h4 class="lead font-pulpRegular">Apartamento en Bella Vista, Santo Domingo</h4>
    <p class="lead font-pulpBold text-primary-100">US$185,000</p>
    <span class="label medium-label">Bella Vista</span>
    <span>3 hab</span>
    <span>120 m²</span>
  </a>
</li>
"""

HOUSE_CARD_HTML = """
<li class="featured-property">
  <a class="property" href="/propiedad/casa-piantini-456">
    <h4 class="lead font-pulpRegular">Casa en Piantini, Santo Domingo</h4>
    <p class="lead font-pulpBold text-primary-100">US$350,000</p>
    <span class="label medium-label">Piantini</span>
    <span>4 hab</span>
    <span>200 m²</span>
  </a>
</li>
"""

# RD$ listing — should be skipped
RD_CARD_HTML = """
<li class="featured-property">
  <a class="property" href="/propiedad/apartamento-naco-789">
    <h4 class="lead font-pulpRegular">Apartamento en Naco, Santo Domingo</h4>
    <p class="lead font-pulpBold text-primary-100">RD$9,500,000</p>
    <span class="label medium-label">Naco</span>
    <span>2 hab</span>
    <span>90 m²</span>
  </a>
</li>
"""

# Missing bedrooms and area — should be skipped
INCOMPLETE_CARD_HTML = """
<li class="featured-property">
  <a class="property" href="/propiedad/apartamento-gazcue-000">
    <h4 class="lead font-pulpRegular">Apartamento en Gazcue, Santo Domingo</h4>
    <p class="lead font-pulpBold text-primary-100">US$120,000</p>
    <span class="label medium-label">Gazcue</span>
  </a>
</li>
"""

# Card with sector derived from title (no span.label.medium-label)
SECTOR_FROM_TITLE_CARD_HTML = """
<li class="featured-property">
  <a class="property" href="/propiedad/apartamento-evaristo-morales-111">
    <h4 class="lead font-pulpRegular">Apartamento en Evaristo Morales, Santo Domingo</h4>
    <p class="lead font-pulpBold text-primary-100">US$230,000</p>
    <span>2 hab</span>
    <span>95 m²</span>
  </a>
</li>
"""

PAGE_WITH_ONE_CARD = f'<html><body><ul>{USD_CARD_HTML}</ul></body></html>'
PAGE_WITH_RD_CARD = f'<html><body><ul>{RD_CARD_HTML}</ul></body></html>'
PAGE_WITH_INCOMPLETE_CARD = f'<html><body><ul>{INCOMPLETE_CARD_HTML}</ul></body></html>'
PAGE_EMPTY = '<html><body><p>no listings</p></body></html>'
PAGE_TWO_CARDS = f'<html><body><ul>{USD_CARD_HTML}{HOUSE_CARD_HTML}</ul></body></html>'


def _mock_response(html: str) -> MagicMock:
    mock = MagicMock()
    mock.text = html
    mock.raise_for_status.return_value = None
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_scrape_returns_list():
    with patch('scraper.sites.plusval.requests.get',
               return_value=_mock_response(PAGE_WITH_ONE_CARD)):
        result = scrape(max_pages=1)
    assert isinstance(result, list)


def test_scrape_each_listing_has_required_keys():
    with patch('scraper.sites.plusval.requests.get',
               return_value=_mock_response(PAGE_WITH_ONE_CARD)):
        result = scrape(max_pages=1)
    assert len(result) > 0
    required_keys = {'price', 'sector', 'property_type', 'bedrooms', 'area_m2', 'source_url'}
    for listing in result:
        assert set(listing.keys()) == required_keys


def test_scrape_source_url_is_absolute():
    with patch('scraper.sites.plusval.requests.get',
               return_value=_mock_response(PAGE_WITH_ONE_CARD)):
        result = scrape(max_pages=1)
    assert len(result) > 0
    for listing in result:
        assert listing['source_url'].startswith('https://')


def test_scrape_skips_non_usd_listings():
    with patch('scraper.sites.plusval.requests.get',
               return_value=_mock_response(PAGE_WITH_RD_CARD)):
        result = scrape(max_pages=1)
    assert result == []


def test_scrape_skips_listing_with_missing_field():
    with patch('scraper.sites.plusval.requests.get',
               return_value=_mock_response(PAGE_WITH_INCOMPLETE_CARD)):
        result = scrape(max_pages=1)
    assert result == []


def test_scrape_empty_page_stops_pagination():
    responses = [
        _mock_response(PAGE_WITH_ONE_CARD),
        _mock_response(PAGE_EMPTY),
    ]
    with patch('scraper.sites.plusval.requests.get', side_effect=responses):
        result = scrape(max_pages=5)
    # Only the first page contributed a listing
    assert len(result) == 1


def test_scrape_property_type_apartment():
    with patch('scraper.sites.plusval.requests.get',
               return_value=_mock_response(PAGE_WITH_ONE_CARD)):
        result = scrape(max_pages=1)
    assert result[0]['property_type'] == 'apartment'


def test_scrape_property_type_house():
    page = f'<html><body><ul>{HOUSE_CARD_HTML}</ul></body></html>'
    with patch('scraper.sites.plusval.requests.get',
               return_value=_mock_response(page)):
        result = scrape(max_pages=1)
    assert result[0]['property_type'] == 'house'


def test_scrape_sector_from_label():
    with patch('scraper.sites.plusval.requests.get',
               return_value=_mock_response(PAGE_WITH_ONE_CARD)):
        result = scrape(max_pages=1)
    assert result[0]['sector'] == 'Bella Vista'


def test_scrape_sector_from_title_fallback():
    page = f'<html><body><ul>{SECTOR_FROM_TITLE_CARD_HTML}</ul></body></html>'
    with patch('scraper.sites.plusval.requests.get',
               return_value=_mock_response(page)):
        result = scrape(max_pages=1)
    assert result[0]['sector'] == 'Santo Domingo'


def test_scrape_price_parsed_correctly():
    with patch('scraper.sites.plusval.requests.get',
               return_value=_mock_response(PAGE_WITH_ONE_CARD)):
        result = scrape(max_pages=1)
    assert result[0]['price'] == 185000
