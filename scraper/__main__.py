"""scraper package entry point.

Usage:
    python -m scraper            # scrape all registered sites into data/listings.db
    python -m scraper export     # export DB to data/listings.csv
"""
import csv
import json
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import scraper.sites.corotos as corotos
import scraper.sites.supercasas as supercasas
import scraper.sites.miscasasrd as miscasasrd
import scraper.sites.mercadolibre as mercadolibre
import scraper.sites.plusval as plusval
import scraper.sites.indominicana as indominicana
import scraper.db as db

# ---------------------------------------------------------------------------
# Registry — hardcoded explicit imports, one per Phase 1 site.
# To add a site: import its module above and add it to REGISTRY.
# Phase 2 / Blocked sites are listed as comments with reason.
# ---------------------------------------------------------------------------
REGISTRY: dict[str, ModuleType] = {
    'corotos': corotos,
    'supercasas': supercasas,
    'miscasasrd': miscasasrd,
    'mercadolibre': mercadolibre,
    'plusval': plusval,
    'indominicana': indominicana,
    # 'casaspb': casaspb,      # Phase 1 — deferred (EasyBroker clone of miscasasrd)
    # 'remaxrd': None,         # Phase 2 — Next.js, JS rendering required
    # 'apartamentosrd': None,  # Phase 2 — Domiclick platform, JS rendering required
    # 'tucasard': None,        # Phase 2 — Domiclick platform (same as apartamentosrd)
}

# ---------------------------------------------------------------------------
# Price bounds for export filtering — matches ml/prepare.py constants
# ---------------------------------------------------------------------------
PRICE_MIN = 10_000   # matches ml/prepare.py PRICE_MIN
PRICE_MAX = 5_000_000  # matches ml/prepare.py PRICE_MAX

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DB_PATH = 'data/listings.db'
CSV_PATH = 'data/listings.csv'
STATUS_PATH = 'data/scrape_status.json'

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def cmd_scrape(db_path: str = DB_PATH, status_path: str = STATUS_PATH) -> None:
    """Scrape all registered Phase 1 sites and insert listings into the DB."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = db.init_db(db_path)

    sites_attempted = len(REGISTRY)
    sites_succeeded = 0
    sites_failed = 0
    failed_sites = []

    for site_name, site_module in REGISTRY.items():
        try:
            listings = site_module.scrape()
            new, skipped = db.insert_listings(conn, listings)
            print(f'{site_name}: {new} new, {skipped} skipped (dedup)')
            sites_succeeded += 1
        except Exception as exc:
            logger.error('%s: failed — %s', site_name, exc)
            sites_failed += 1
            failed_sites.append(site_name)

    conn.close()

    print(f'Scraped {sites_succeeded} of {sites_attempted} sites; {sites_failed} failed.')

    # Write scrape status sidecar for cmd_export to read
    status = {
        'sites_attempted': sites_attempted,
        'sites_succeeded': sites_succeeded,
        'sites_failed': sites_failed,
        'failed_sites': failed_sites,
        'run_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    }
    Path(status_path).parent.mkdir(parents=True, exist_ok=True)
    with open(status_path, 'w') as f:
        json.dump(status, f)


def cmd_export(
    db_path: str = DB_PATH,
    csv_path: str = CSV_PATH,
    status_path: str = STATUS_PATH,
) -> None:
    """Export clean listings from the DB to data/listings.csv.

    Applies PRICE_MIN / PRICE_MAX bounds. Strips whitespace from sector
    (no title-casing — preserves the casing the encoder was trained on).
    Exits non-zero if no rows pass cleaning.
    Warns if the last scrape had site failures.
    """
    # Warn if last scrape had failures
    if Path(status_path).exists():
        try:
            with open(status_path) as f:
                status = json.load(f)
        except json.JSONDecodeError:
            print('WARNING: scrape_status.json is unreadable — export may be incomplete.', file=sys.stderr)
            status = {}
        if status.get('sites_failed', 0) > 0:
            num_failed = status['sites_failed']
            names = ', '.join(status.get('failed_sites', []))
            print(
                f'WARNING: last scrape had {num_failed} site failure(s) ({names}) — '
                'export may be incomplete. Review scrape logs before retraining.',
                file=sys.stderr,
            )

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            'SELECT price, sector, property_type, bedrooms, area_m2 '
            'FROM listings WHERE price IS NOT NULL AND area_m2 IS NOT NULL'
        ).fetchall()
    finally:
        conn.close()

    # Apply price bounds and sector whitespace strip
    fieldnames = ['price', 'sector', 'property_type', 'bedrooms', 'area_m2']
    clean = []
    for price, sector, property_type, bedrooms, area_m2 in rows:
        if PRICE_MIN <= price <= PRICE_MAX:
            clean.append({
                'price': price,
                'sector': sector.strip() if sector else sector,
                'property_type': property_type,
                'bedrooms': bedrooms,
                'area_m2': area_m2,
            })

    if not clean:
        print(
            'Error: no rows pass price bounds '
            f'(PRICE_MIN={PRICE_MIN}, PRICE_MAX={PRICE_MAX}). '
            'Run the scraper first.',
            file=sys.stderr,
        )
        sys.exit(1)

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clean)

    print(f'Exported {len(clean)} rows to {csv_path}')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'export':
        cmd_export()
    else:
        cmd_scrape()
