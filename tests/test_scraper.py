from bs4 import BeautifulSoup
from scraper.scraper import parse_listing, parse_price_usd

# Minimal fixture matching the CARD_SELECTOR, PRICE_CURRENCY_SELECTOR, etc. in scraper.py
# Structure mirrors the real corotos.com.do HTML (verified by live inspection)
SAMPLE_CARD_HTML = """
<div class="listing-item col s12 l4">
  <div class="container_search">
    <div class="card-item listing__item">
      <div class="bottom-card-content">
        <div class="card-info">
          <div class="card-info-top">
            <a href="/anuncio/test">
              <h3>Apartamento en Venta en Piantini</h3>
            </a>
            <a class="item__price" href="/anuncio/test">
              <div class="card-row">
                <span class="item__currency">US$</span>
                <span class="item__price-amount">185,000</span>
              </div>
            </a>
          </div>
          <div class="card-info-bottom">
            <a class="info-wrapper" href="/anuncio/test">
              <div class="location-more">
                <span class="cards-subtitles-gray">Venta</span>
              </div>
              <div class="location-more">
                <span class="listing-text-gray card-right-text">Piantini</span>
              </div>
              <div class="real-estate-more">
                <span>3 habs</span>
                <span class="margin-text">·</span>
                <span>120 m²</span>
              </div>
            </a>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
"""


def card():
    return BeautifulSoup(SAMPLE_CARD_HTML, 'html.parser').select_one('div.listing-item')


def test_parse_price_usd_valid():
    assert parse_price_usd("US$185,000") == 185000


def test_parse_price_usd_rejects_peso():
    assert parse_price_usd("RD$3,500,000") is None


def test_parse_listing_price():
    assert parse_listing(card())['price'] == 185000


def test_parse_listing_sector():
    assert parse_listing(card())['sector'] == 'Piantini'


def test_parse_listing_area():
    assert parse_listing(card())['area_m2'] == 120


def test_parse_listing_property_type():
    assert parse_listing(card())['property_type'] == 'apartment'


def test_parse_listing_bedrooms():
    assert parse_listing(card())['bedrooms'] == 3
