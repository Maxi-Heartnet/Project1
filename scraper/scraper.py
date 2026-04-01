import csv
import os
import re
import time

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.corotos.com.do/sc/inmuebles/apartamentos"
RAW_DIR = "data/raw"
OUTPUT_CSV = "data/listings.csv"
FIELDNAMES = ['price', 'sector', 'property_type', 'bedrooms', 'bathrooms',
              'area_m2', 'parking', 'floor_level']

# CSS selectors — verified against live site (corotos.com.do)
CARD_SELECTOR = "div.listing-item"
PRICE_CURRENCY_SELECTOR = "span.item__currency"
PRICE_AMOUNT_SELECTOR = "span.item__price-amount"
TITLE_SELECTOR = "h3"
DETAILS_SELECTOR = "div.real-estate-more span"
LOCATION_SELECTOR = "span.listing-text-gray"


def parse_price_usd(text):
    """Return integer USD price, or None if not a USD listing."""
    text = text.strip()
    upper = text.upper()
    if 'RD$' in upper or 'DOP' in upper:
        return None
    if 'US$' not in upper and 'USD' not in upper and '$' not in text:
        return None
    numbers = re.findall(r'\d+', text)
    return int(''.join(numbers)) if numbers else None


def _first_int(text):
    numbers = re.findall(r'\d+', text)
    return int(numbers[0]) if numbers else None


def parse_listing(card):
    """Parse a BeautifulSoup Tag (one listing card) into a dict."""
    currency_el = card.select_one(PRICE_CURRENCY_SELECTOR)
    amount_el = card.select_one(PRICE_AMOUNT_SELECTOR)
    if currency_el and amount_el:
        price_text = currency_el.get_text(strip=True) + amount_el.get_text(strip=True)
        price = parse_price_usd(price_text)
    else:
        price = None

    location_el = card.select_one(LOCATION_SELECTOR)
    sector = location_el.get_text(strip=True) if location_el else None

    title_el = card.select_one(TITLE_SELECTOR)
    title = title_el.get_text(strip=True).lower() if title_el else ''
    if 'apartamento' in title or 'apartment' in title:
        property_type = 'apartment'
    elif 'casa' in title or 'house' in title:
        property_type = 'house'
    else:
        property_type = 'apartment'

    details = [el.get_text(strip=True) for el in card.select(DETAILS_SELECTOR)]
    bedrooms = bathrooms = area_m2 = parking = floor_level = None
    for d in details:
        dl = d.lower()
        if bedrooms is None and ('hab' in dl or 'bedroom' in dl):
            bedrooms = _first_int(d)
        elif bathrooms is None and ('baño' in dl or 'bathroom' in dl or 'bano' in dl):
            bathrooms = _first_int(d)
        elif area_m2 is None and ('m²' in d or 'm2' in dl or 'm�' in d):
            area_m2 = _first_int(d)
        elif parking is None and ('parking' in dl or 'garage' in dl or 'garaje' in dl):
            parking = _first_int(d)
        elif floor_level is None and ('piso' in dl or 'floor' in dl):
            floor_level = _first_int(d)

    return {
        'price': price, 'sector': sector, 'property_type': property_type,
        'bedrooms': bedrooms, 'bathrooms': bathrooms, 'area_m2': area_m2,
        'parking': parking, 'floor_level': floor_level,
    }


def fetch_page(page_num, use_cache=True):
    cache_path = os.path.join(RAW_DIR, f"page_{page_num}.html")
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return f.read()
    resp = requests.get(
        BASE_URL,
        params={'page': page_num},
        headers={'User-Agent': 'Mozilla/5.0'},
        timeout=15,
    )
    resp.raise_for_status()
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(cache_path, 'w', encoding='utf-8') as f:
        f.write(resp.text)
    return resp.text


def scrape(max_pages=30, use_cache=True):
    listings = []
    for page_num in range(1, max_pages + 1):
        html = fetch_page(page_num, use_cache=use_cache)
        soup = BeautifulSoup(html, 'html.parser')
        cards = soup.select(CARD_SELECTOR)
        if not cards:
            print(f"Page {page_num}: no cards found — stopping.")
            break
        for card in cards:
            row = parse_listing(card)
            if row['price'] and row['area_m2']:
                listings.append(row)
        print(f"Page {page_num}: {len(cards)} cards | total so far: {len(listings)}")
        time.sleep(1)
    return listings


def save_csv(listings, path=OUTPUT_CSV):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(listings)
    print(f"Saved {len(listings)} listings to {path}")


if __name__ == '__main__':
    save_csv(scrape())
