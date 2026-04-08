"""Tests for scraper/sites/mercadolibre.py.

All tests mock requests.get so no real network calls are made.
"""
from unittest.mock import patch, MagicMock

from scraper.sites.mercadolibre import scrape

# ---------------------------------------------------------------------------
# HTML fixtures
# ---------------------------------------------------------------------------

USD_CARD_HTML = """
<div class="ui-search-result__wrapper">
  <a class="poly-component__title"
     href="https://inmuebles.mercadolibre.com.do/apartamento/apartamento-en-bella-vista-123">
    Apartamento en Bella Vista
  </a>
  <span class="andes-money-amount__currency-symbol">$</span>
  <span class="andes-money-amount__fraction">185,000</span>
  <span class="poly-component__location">Bella Vista, Santo Domingo</span>
  <span>3 hab</span>
  <span>120 m²</span>
</div>
"""

HOUSE_CARD_HTML = """
<div class="ui-search-result__wrapper">
  <a class="poly-component__title"
     href="https://inmuebles.mercadolibre.com.do/casas/casa-en-piantini-456">
    Casa en Piantini
  </a>
  <span class="andes-money-amount__currency-symbol">$</span>
  <span class="andes-money-amount__fraction">350,000</span>
  <span class="poly-component__location">Piantini, Santo Domingo</span>
  <span>4 hab</span>
  <span>200 m²</span>
</div>
"""

# RD$ listing — should be skipped
RD_CARD_HTML = """
<div class="ui-search-result__wrapper">
  <a class="poly-component__title"
     href="https://inmuebles.mercadolibre.com.do/apartamento/ap-rd-789">
    Apartamento en Naco
  </a>
  <span class="andes-money-amount__currency-symbol">$</span>
  <span class="andes-money-amount__fraction">9,500,000</span>
  <span class="poly-component__location">Naco, Santo Domingo</span>
  <span>RD$ 9,500,000</span>
  <span>2 hab</span>
  <span>90 m²</span>
</div>
"""

# Missing bedrooms and area — should be skipped
INCOMPLETE_CARD_HTML = """
<div class="ui-search-result__wrapper">
  <a class="poly-component__title"
     href="https://inmuebles.mercadolibre.com.do/apartamento/ap-incomplete-000">
    Apartamento en Gazcue
  </a>
  <span class="andes-money-amount__currency-symbol">$</span>
  <span class="andes-money-amount__fraction">120,000</span>
  <span class="poly-component__location">Gazcue, Santo Domingo</span>
</div>
"""

PAGE_WITH_ONE_CARD = f'<html><body>{USD_CARD_HTML}</body></html>'
PAGE_WITH_RD_CARD = f'<html><body>{RD_CARD_HTML}</body></html>'
PAGE_WITH_INCOMPLETE_CARD = f'<html><body>{INCOMPLETE_CARD_HTML}</body></html>'
PAGE_EMPTY = '<html><body><p>no listings</p></body></html>'
PAGE_TWO_CARDS = f'<html><body>{USD_CARD_HTML}{HOUSE_CARD_HTML}</body></html>'


def _mock_response(html: str) -> MagicMock:
    mock = MagicMock()
    mock.text = html
    mock.raise_for_status.return_value = None
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_scrape_returns_list():
    with patch('scraper.sites.mercadolibre.requests.get',
               return_value=_mock_response(PAGE_WITH_ONE_CARD)):
        result = scrape(max_pages=1)
    assert isinstance(result, list)


def test_scrape_each_listing_has_required_keys():
    with patch('scraper.sites.mercadolibre.requests.get',
               return_value=_mock_response(PAGE_WITH_ONE_CARD)):
        result = scrape(max_pages=1)
    assert len(result) > 0
    required_keys = {'price', 'sector', 'property_type', 'bedrooms', 'area_m2', 'source_url'}
    for listing in result:
        assert set(listing.keys()) == required_keys


def test_scrape_source_url_is_absolute():
    with patch('scraper.sites.mercadolibre.requests.get',
               return_value=_mock_response(PAGE_WITH_ONE_CARD)):
        result = scrape(max_pages=1)
    assert len(result) > 0
    for listing in result:
        assert listing['source_url'].startswith('https://')


def test_scrape_skips_non_usd_listings():
    with patch('scraper.sites.mercadolibre.requests.get',
               return_value=_mock_response(PAGE_WITH_RD_CARD)):
        result = scrape(max_pages=1)
    assert result == []


def test_scrape_skips_listing_with_missing_field():
    with patch('scraper.sites.mercadolibre.requests.get',
               return_value=_mock_response(PAGE_WITH_INCOMPLETE_CARD)):
        result = scrape(max_pages=1)
    assert result == []


def test_scrape_empty_page_stops_pagination():
    responses = [
        _mock_response(PAGE_WITH_ONE_CARD),
        _mock_response(PAGE_EMPTY),
    ]
    with patch('scraper.sites.mercadolibre.requests.get', side_effect=responses):
        result = scrape(max_pages=5)
    # Only the first page contributed a listing
    assert len(result) == 1


def test_scrape_property_type_apartment():
    with patch('scraper.sites.mercadolibre.requests.get',
               return_value=_mock_response(PAGE_WITH_ONE_CARD)):
        result = scrape(max_pages=1)
    assert result[0]['property_type'] == 'apartment'


def test_scrape_property_type_house():
    page = f'<html><body>{HOUSE_CARD_HTML}</body></html>'
    with patch('scraper.sites.mercadolibre.requests.get',
               return_value=_mock_response(page)):
        result = scrape(max_pages=1)
    assert result[0]['property_type'] == 'house'


def test_scrape_sector_is_first_part_before_comma():
    with patch('scraper.sites.mercadolibre.requests.get',
               return_value=_mock_response(PAGE_WITH_ONE_CARD)):
        result = scrape(max_pages=1)
    assert result[0]['sector'] == 'Bella Vista'


def test_scrape_price_parsed_correctly():
    with patch('scraper.sites.mercadolibre.requests.get',
               return_value=_mock_response(PAGE_WITH_ONE_CARD)):
        result = scrape(max_pages=1)
    assert result[0]['price'] == 185000
