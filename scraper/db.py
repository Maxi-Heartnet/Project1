import sqlite3
from datetime import datetime, timezone


REQUIRED_KEYS = {'price', 'sector', 'property_type', 'bedrooms', 'area_m2', 'source_url'}


def init_db(db_path: str) -> sqlite3.Connection:
    """Create the listings table if it doesn't exist and return an open connection."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS listings (
            id          INTEGER PRIMARY KEY,
            price       INTEGER,
            sector      TEXT,
            property_type TEXT,
            bedrooms    INTEGER,
            area_m2     REAL,
            source_url  TEXT UNIQUE,
            scraped_at  TEXT
        )
    """)
    conn.commit()
    return conn


def insert_listings(conn: sqlite3.Connection, listings: list) -> tuple[int, int]:
    """Insert listings with URL-based deduplication.

    Validates that each listing dict contains exactly REQUIRED_KEYS.
    Inserts row-by-row using INSERT OR IGNORE, tracking counts reliably.
    Sectors are stored as-is (no casing normalization).

    Returns (new_count, skipped_count).
    """
    new_count = 0
    skipped_count = 0
    scraped_at = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    for listing in listings:
        missing = REQUIRED_KEYS - set(listing.keys())
        extra = set(listing.keys()) - REQUIRED_KEYS
        if missing or extra:
            raise ValueError(
                f"Listing dict has wrong keys. "
                f"Missing: {sorted(missing) or 'none'}. "
                f"Extra: {sorted(extra) or 'none'}."
            )

        cur = conn.execute(
            """INSERT OR IGNORE INTO listings
               (price, sector, property_type, bedrooms, area_m2, source_url, scraped_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                listing['price'],
                listing['sector'],
                listing['property_type'],
                listing['bedrooms'],
                listing['area_m2'],
                listing['source_url'],
                scraped_at,
            ),
        )
        if cur.rowcount == 1:
            new_count += 1
        else:
            skipped_count += 1

    conn.commit()
    return new_count, skipped_count
