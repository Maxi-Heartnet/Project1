"""
Generate static/sector_coords.json — the sector-name → lat/lng lookup for map pins.

Usage:
    GEOCODING_API_KEY=<key> python scripts/generate_sector_coords.py

Requirements:
    - ml/model.pkl must be present locally (run ml/train.py first if absent).
    - GEOCODING_API_KEY env var must be set to a Google API key with Geocoding API enabled.
      This key is used ONCE and should be revoked immediately after the script completes.
      Do NOT store it in .env files or shell history; pass it inline as shown above.

Output:
    static/sector_coords.json — dict of { "SectorName": { "lat": float, "lng": float } }
    Sectors with no resolvable coordinate within Dominican Republic bounds are omitted.

DR lat/lng bounds used for validation:
    lat: 17.4 – 19.9  (all positive)
    lng: -74.5 – -68.3  (all negative)
"""

import json
import logging
import os
import sys
from pathlib import Path

import joblib
import requests

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent / 'ml' / 'model.pkl'
OUTPUT_PATH = Path(__file__).parent.parent / 'static' / 'sector_coords.json'

GEOCODING_URL = 'https://maps.googleapis.com/maps/api/geocode/json'

# Dominican Republic geographic bounds (all longitudes are negative)
LAT_MIN, LAT_MAX = 17.4, 19.9
LNG_MIN, LNG_MAX = -74.5, -68.3


def load_sector_names() -> list[str]:
    if not MODEL_PATH.exists():
        logger.error('Model not found at %s — run ml/train.py first.', MODEL_PATH)
        sys.exit(1)
    artifact = joblib.load(MODEL_PATH)
    try:
        sectors = (
            artifact['encoder']
            .named_transformers_['cat']
            .categories_[0]
            .tolist()
        )
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        logger.error('Could not extract sector list from encoder: %s', exc)
        sys.exit(1)
    logger.info('Loaded %d sectors from model.pkl', len(sectors))
    return sectors


def geocode(sector_name: str, api_key: str) -> dict | None:
    query = f'{sector_name}, Santo Domingo, Dominican Republic'
    try:
        resp = requests.get(
            GEOCODING_URL,
            params={'address': query, 'key': api_key},
            timeout=10,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning('HTTP error geocoding "%s": %s', sector_name, exc)
        return None

    data = resp.json()
    if data.get('status') != 'OK' or not data.get('results'):
        logger.warning('No results for "%s" (status: %s)', sector_name, data.get('status'))
        return None

    loc = data['results'][0]['geometry']['location']
    lat, lng = loc['lat'], loc['lng']

    if not (LAT_MIN <= lat <= LAT_MAX and LNG_MIN <= lng <= LNG_MAX):
        logger.warning(
            'Coordinate for "%s" (%.4f, %.4f) is outside DR bounds — omitting.',
            sector_name, lat, lng,
        )
        return None

    return {'lat': lat, 'lng': lng}


def main():
    api_key = os.environ.get('GEOCODING_API_KEY', '')
    if not api_key:
        logger.error('GEOCODING_API_KEY env var is not set.')
        sys.exit(1)

    sectors = load_sector_names()
    coords: dict[str, dict] = {}

    for i, sector in enumerate(sectors, 1):
        logger.info('[%d/%d] Geocoding "%s"…', i, len(sectors), sector)
        result = geocode(sector, api_key)
        if result:
            coords[sector] = result

    omitted = len(sectors) - len(coords)
    coverage = len(coords) / len(sectors) if sectors else 0
    logger.info(
        'Done: %d/%d sectors resolved (%.0f%% coverage), %d omitted.',
        len(coords), len(sectors), coverage * 100, omitted,
    )
    if coverage < 0.8:
        logger.warning(
            'Coverage is below 80%% (%d/%d). Check warnings above.',
            len(coords), len(sectors),
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(coords, f, indent=2, ensure_ascii=False)
    logger.info('Written to %s', OUTPUT_PATH)


if __name__ == '__main__':
    main()
