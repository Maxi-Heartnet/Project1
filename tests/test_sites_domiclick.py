"""Tests for scraper/sites/_domiclick.py — parse_listings() tested directly, no Playwright.

Also covers apartamentosrd.py and tucasard.py wrappers via integration smoke tests.
"""
import pytest

from scraper.sites._domiclick import parse_listings

APARTAMENTOSRD_BASE = 'https://www.apartamentosrd.com.do'
TUCASARD_BASE = 'https://www.tucasard.com'

# ---------------------------------------------------------------------------
# HTML fixtures
# ---------------------------------------------------------------------------
# Matches Domiclick DOM as spiked on 2026-04-10:
# - Cards: div.card.h-100
# - Anchor: a[href*="/propiedad/"] inside card
# - Property type: h5 inside div.property-content
# - Sector: span with i.fas.fa-map-marker-alt — "Serrallés, Santo Domingo D.N."
# - Price: span — "US$ 165,000" or "RD$ X,XXX,XXX"
# - Bedrooms: li with i.fas.fa-bed — "1 Hab." or "Desde 1 hasta 3 Hab."
# - Area: li with i.fas.fa-arrows-alt — "58 Mt2"

USD_CARD = """
<div class="card h-100">
  <a href="/propiedad/12345/">
    <div class="property-content">
      <h5>Apartamento en Venta</h5>
      <span><i class="fas fa-map-marker-alt"></i>Serrallés, Santo Domingo D.N.</span>
    </div>
    <ul>
      <li><span>US$ 165,000</span></li>
    </ul>
    <ul class="features">
      <li><i class="fas fa-bed"></i>1 Hab.</li>
      <li><i class="fas fa-arrows-alt"></i>58 Mt2</li>
    </ul>
  </a>
</div>
"""

VILLA_CARD = """
<div class="card h-100">
  <a href="/propiedad/99999/">
    <div class="property-content">
      <h5>Villa en Venta</h5>
      <span><i class="fas fa-map-marker-alt"></i>Punta Cana, La Altagracia</span>
    </div>
    <ul>
      <li><span>US$ 850,000</span></li>
    </ul>
    <ul class="features">
      <li><i class="fas fa-bed"></i>4 Hab.</li>
      <li><i class="fas fa-arrows-alt"></i>320 Mt2</li>
    </ul>
  </a>
</div>
"""

DOP_CARD = """
<div class="card h-100">
  <a href="/propiedad/11111/">
    <div class="property-content">
      <h5>Apartamento en Venta</h5>
      <span><i class="fas fa-map-marker-alt"></i>Piantini, Santo Domingo</span>
    </div>
    <ul>
      <li><span>RD$ 9,500,000</span></li>
    </ul>
    <ul class="features">
      <li><i class="fas fa-bed"></i>2 Hab.</li>
      <li><i class="fas fa-arrows-alt"></i>80 Mt2</li>
    </ul>
  </a>
</div>
"""

NO_BEDROOMS_CARD = """
<div class="card h-100">
  <a href="/propiedad/22222/">
    <div class="property-content">
      <h5>Apartamento en Venta</h5>
      <span><i class="fas fa-map-marker-alt"></i>Gazcue, Santo Domingo</span>
    </div>
    <ul>
      <li><span>US$ 220,000</span></li>
    </ul>
    <ul class="features">
      <li><i class="fas fa-arrows-alt"></i>95 Mt2</li>
    </ul>
  </a>
</div>
"""

NO_AREA_CARD = """
<div class="card h-100">
  <a href="/propiedad/33333/">
    <div class="property-content">
      <h5>Apartamento en Venta</h5>
      <span><i class="fas fa-map-marker-alt"></i>Naco, Santo Domingo</span>
    </div>
    <ul>
      <li><span>US$ 250,000</span></li>
    </ul>
    <ul class="features">
      <li><i class="fas fa-bed"></i>2 Hab.</li>
    </ul>
  </a>
</div>
"""

RANGE_BEDS_CARD = """
<div class="card h-100">
  <a href="/propiedad/44444/">
    <div class="property-content">
      <h5>Apartamento en Venta</h5>
      <span><i class="fas fa-map-marker-alt"></i>Bella Vista, Santo Domingo</span>
    </div>
    <ul>
      <li><span>US$ 300,000</span></li>
    </ul>
    <ul class="features">
      <li><i class="fas fa-bed"></i>Desde 1 hasta 3 Hab.</li>
      <li><i class="fas fa-arrows-alt"></i>120 Mt2</li>
    </ul>
  </a>
</div>
"""


def _page(cards_html):
    return '<html><body>{}</body></html>'.format(cards_html)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDomiclickParseListings:

    def test_returns_list(self):
        result = parse_listings(_page(USD_CARD), APARTAMENTOSRD_BASE)
        assert isinstance(result, list)

    def test_each_listing_has_required_keys(self):
        result = parse_listings(_page(USD_CARD), APARTAMENTOSRD_BASE)
        assert len(result) == 1
        required = {'price', 'sector', 'property_type', 'bedrooms', 'area_m2', 'source_url'}
        assert set(result[0].keys()) == required

    def test_source_url_absolute_with_base(self):
        result = parse_listings(_page(USD_CARD), APARTAMENTOSRD_BASE)
        assert result[0]['source_url'].startswith(APARTAMENTOSRD_BASE)
        assert '/propiedad/12345/' in result[0]['source_url']

    def test_source_url_uses_tucasard_base(self):
        result = parse_listings(_page(USD_CARD), TUCASARD_BASE)
        assert result[0]['source_url'].startswith(TUCASARD_BASE)

    def test_price_parsed_correctly(self):
        result = parse_listings(_page(USD_CARD), APARTAMENTOSRD_BASE)
        assert result[0]['price'] == 165000

    def test_sector_first_segment(self):
        """'Serrallés, Santo Domingo D.N.' → 'Serrallés'"""
        result = parse_listings(_page(USD_CARD), APARTAMENTOSRD_BASE)
        assert result[0]['sector'] == 'Serrallés'

    def test_bedrooms_parsed(self):
        result = parse_listings(_page(USD_CARD), APARTAMENTOSRD_BASE)
        assert result[0]['bedrooms'] == 1

    def test_area_parsed(self):
        result = parse_listings(_page(USD_CARD), APARTAMENTOSRD_BASE)
        assert result[0]['area_m2'] == 58.0

    def test_villa_property_type(self):
        result = parse_listings(_page(VILLA_CARD), APARTAMENTOSRD_BASE)
        assert result[0]['property_type'] == 'house'

    def test_apartment_property_type(self):
        result = parse_listings(_page(USD_CARD), APARTAMENTOSRD_BASE)
        assert result[0]['property_type'] == 'apartment'

    def test_skips_dop_listings(self):
        result = parse_listings(_page(DOP_CARD), APARTAMENTOSRD_BASE)
        assert result == []

    def test_skips_missing_bedrooms(self):
        result = parse_listings(_page(NO_BEDROOMS_CARD), APARTAMENTOSRD_BASE)
        assert result == []

    def test_skips_missing_area(self):
        result = parse_listings(_page(NO_AREA_CARD), APARTAMENTOSRD_BASE)
        assert result == []

    def test_range_beds_extracts_first_integer(self):
        """'Desde 1 hasta 3 Hab.' → bedrooms=1"""
        result = parse_listings(_page(RANGE_BEDS_CARD), APARTAMENTOSRD_BASE)
        assert len(result) == 1
        assert result[0]['bedrooms'] == 1

    def test_empty_page_returns_empty_list(self):
        result = parse_listings('<html><body></body></html>', APARTAMENTOSRD_BASE)
        assert result == []

    def test_multiple_cards(self):
        result = parse_listings(_page(USD_CARD + VILLA_CARD), APARTAMENTOSRD_BASE)
        assert len(result) == 2

    def test_dop_filtered_from_mixed_page(self):
        """USD card + DOP card → only 1 result."""
        result = parse_listings(_page(USD_CARD + DOP_CARD), APARTAMENTOSRD_BASE)
        assert len(result) == 1
        assert result[0]['price'] == 165000
