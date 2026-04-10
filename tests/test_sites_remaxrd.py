"""Tests for scraper/sites/remaxrd.py — all Playwright is bypassed, _parse_listings tested directly."""
import pytest

from scraper.sites.remaxrd import _parse_listings, REMAXRD_BASE

# ---------------------------------------------------------------------------
# HTML fixtures
# ---------------------------------------------------------------------------
# Matches remaxrd.com DOM as spiked on 2026-04-10:
# - Cards are a[href*="/propiedad/"] (anchor wraps entire card)
# - h3 for property type text
# - span sibling after h3 for sector ("ENSANCHE NACO, SANTO DOMINGO DE GUZMÁN")
# - span containing "US$" for price
# - li with img[src*="icon_bed"] for bedrooms
# - li with img[src*="icon_rule"] for area ("233.60 M2")

USD_CARD = """
<a href="/propiedad/12345/" target="_blank">
  <h3>Apartamento en Venta</h3>
  <span>ENSANCHE NACO, SANTO DOMINGO DE GUZMÁN</span>
  <div class="price-container">
    <span>US$565,000</span>
  </div>
  <ul>
    <li><img src="/icons/icon_bed_remaxrd.svg" alt="beds" /> 3</li>
    <li><img src="/icons/icon_rule_remaxrd.svg" alt="area" /><span>233.60 M2</span></li>
  </ul>
</a>
"""

VILLA_CARD = """
<a href="/propiedad/99999/" target="_blank">
  <h3>Villa en Venta</h3>
  <span>PUNTA CANA, LA ALTAGRACIA</span>
  <div class="price-container">
    <span>US$1,200,000</span>
  </div>
  <ul>
    <li><img src="/icons/icon_bed_remaxrd.svg" alt="beds" /> 5</li>
    <li><img src="/icons/icon_rule_remaxrd.svg" alt="area" /><span>450 M2</span></li>
  </ul>
</a>
"""

DOP_CARD = """
<a href="/propiedad/11111/" target="_blank">
  <h3>Apartamento en Venta</h3>
  <span>PIANTINI, SANTO DOMINGO</span>
  <div class="price-container">
    <span>RD$25,000,000</span>
  </div>
  <ul>
    <li><img src="/icons/icon_bed_remaxrd.svg" alt="beds" /> 2</li>
    <li><img src="/icons/icon_rule_remaxrd.svg" alt="area" /><span>100 M2</span></li>
  </ul>
</a>
"""

NO_PRICE_CARD = """
<a href="/propiedad/22222/" target="_blank">
  <h3>Casa en Venta</h3>
  <span>BELLA VISTA, SANTO DOMINGO</span>
  <ul>
    <li><img src="/icons/icon_bed_remaxrd.svg" alt="beds" /> 4</li>
    <li><img src="/icons/icon_rule_remaxrd.svg" alt="area" /><span>300 M2</span></li>
  </ul>
</a>
"""

NO_BEDROOMS_CARD = """
<a href="/propiedad/33333/" target="_blank">
  <h3>Apartamento en Venta</h3>
  <span>GAZCUE, SANTO DOMINGO</span>
  <div class="price-container">
    <span>US$280,000</span>
  </div>
  <ul>
    <li><img src="/icons/icon_rule_remaxrd.svg" alt="area" /><span>120 M2</span></li>
  </ul>
</a>
"""

NO_AREA_CARD = """
<a href="/propiedad/44444/" target="_blank">
  <h3>Apartamento en Venta</h3>
  <span>NACO, SANTO DOMINGO</span>
  <div class="price-container">
    <span>US$320,000</span>
  </div>
  <ul>
    <li><img src="/icons/icon_bed_remaxrd.svg" alt="beds" /> 2</li>
  </ul>
</a>
"""

NO_SECTOR_CARD = """
<a href="/propiedad/55555/" target="_blank">
  <div class="price-container">
    <span>US$400,000</span>
  </div>
  <ul>
    <li><img src="/icons/icon_bed_remaxrd.svg" alt="beds" /> 3</li>
    <li><img src="/icons/icon_rule_remaxrd.svg" alt="area" /><span>150 M2</span></li>
  </ul>
</a>
"""


def _page(cards_html):
    return '<html><body>{}</body></html>'.format(cards_html)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRemaxrdParseListings:

    def test_returns_list(self):
        result = _parse_listings(_page(USD_CARD))
        assert isinstance(result, list)

    def test_each_listing_has_required_keys(self):
        result = _parse_listings(_page(USD_CARD))
        assert len(result) == 1
        required = {'price', 'sector', 'property_type', 'bedrooms', 'area_m2', 'source_url'}
        assert set(result[0].keys()) == required

    def test_source_url_absolute_with_base(self):
        result = _parse_listings(_page(USD_CARD))
        assert result[0]['source_url'].startswith(REMAXRD_BASE)
        assert '/propiedad/12345/' in result[0]['source_url']

    def test_price_parsed_correctly(self):
        result = _parse_listings(_page(USD_CARD))
        assert result[0]['price'] == 565000

    def test_sector_first_segment_title_case(self):
        """'ENSANCHE NACO, SANTO DOMINGO DE GUZMÁN' → 'Ensanche Naco'"""
        result = _parse_listings(_page(USD_CARD))
        assert result[0]['sector'] == 'Ensanche Naco'

    def test_bedrooms_parsed(self):
        result = _parse_listings(_page(USD_CARD))
        assert result[0]['bedrooms'] == 3

    def test_area_parsed(self):
        result = _parse_listings(_page(USD_CARD))
        assert result[0]['area_m2'] == 233.60

    def test_villa_property_type(self):
        result = _parse_listings(_page(VILLA_CARD))
        assert result[0]['property_type'] == 'house'

    def test_apartment_property_type(self):
        result = _parse_listings(_page(USD_CARD))
        assert result[0]['property_type'] == 'apartment'

    def test_skips_dop_listings(self):
        result = _parse_listings(_page(DOP_CARD))
        assert result == []

    def test_skips_missing_price(self):
        result = _parse_listings(_page(NO_PRICE_CARD))
        assert result == []

    def test_skips_missing_bedrooms(self):
        result = _parse_listings(_page(NO_BEDROOMS_CARD))
        assert result == []

    def test_skips_missing_area(self):
        result = _parse_listings(_page(NO_AREA_CARD))
        assert result == []

    def test_skips_missing_sector(self):
        result = _parse_listings(_page(NO_SECTOR_CARD))
        assert result == []

    def test_empty_page_returns_empty_list(self):
        result = _parse_listings('<html><body></body></html>')
        assert result == []

    def test_multiple_cards(self):
        result = _parse_listings(_page(USD_CARD + VILLA_CARD))
        assert len(result) == 2

    def test_dop_filtered_from_mixed_page(self):
        """USD card + DOP card → only 1 result."""
        result = _parse_listings(_page(USD_CARD + DOP_CARD))
        assert len(result) == 1
        assert result[0]['price'] == 565000
