import pytest
from datetime import datetime, timezone
from scraper.db import init_db, insert_listings, REQUIRED_KEYS


SAMPLE_LISTING = {
    'price': 185000,
    'sector': 'Piantini',
    'property_type': 'apartment',
    'bedrooms': 3,
    'area_m2': 120.0,
    'source_url': 'https://corotos.com.do/anuncio/12345',
}


def test_init_db_creates_table(tmp_path):
    conn = init_db(str(tmp_path / 'test.db'))
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='listings'"
    )
    assert cur.fetchone() is not None
    conn.close()


def test_init_db_idempotent(tmp_path):
    db_path = str(tmp_path / 'test.db')
    init_db(db_path).close()
    # Second call should not raise
    conn = init_db(db_path)
    conn.close()


def test_insert_new_listing_returns_1_0(tmp_path):
    conn = init_db(str(tmp_path / 'test.db'))
    result = insert_listings(conn, [SAMPLE_LISTING])
    assert result == (1, 0)
    conn.close()


def test_insert_duplicate_url_returns_0_1(tmp_path):
    conn = init_db(str(tmp_path / 'test.db'))
    insert_listings(conn, [SAMPLE_LISTING])
    result = insert_listings(conn, [SAMPLE_LISTING])
    assert result == (0, 1)
    conn.close()


def test_insert_duplicate_db_row_count_stays_1(tmp_path):
    conn = init_db(str(tmp_path / 'test.db'))
    insert_listings(conn, [SAMPLE_LISTING])
    insert_listings(conn, [SAMPLE_LISTING])
    count = conn.execute('SELECT COUNT(*) FROM listings').fetchone()[0]
    assert count == 1
    conn.close()


def test_insert_invalid_keys_raises_value_error(tmp_path):
    conn = init_db(str(tmp_path / 'test.db'))
    bad = {k: v for k, v in SAMPLE_LISTING.items() if k != 'source_url'}
    with pytest.raises(ValueError, match='source_url'):
        insert_listings(conn, [bad])
    conn.close()


def test_insert_extra_keys_raises_value_error(tmp_path):
    conn = init_db(str(tmp_path / 'test.db'))
    bad = {**SAMPLE_LISTING, 'extra_field': 'oops'}
    with pytest.raises(ValueError):
        insert_listings(conn, [bad])
    conn.close()


def test_insert_stores_sector_as_is(tmp_path):
    conn = init_db(str(tmp_path / 'test.db'))
    listing = {**SAMPLE_LISTING, 'sector': 'piantini', 'source_url': 'https://ex.com/1'}
    insert_listings(conn, [listing])
    row = conn.execute('SELECT sector FROM listings WHERE source_url=?', ('https://ex.com/1',)).fetchone()
    assert row[0] == 'piantini'
    conn.close()


def test_insert_batch_mixed_new_and_duplicate(tmp_path):
    conn = init_db(str(tmp_path / 'test.db'))
    listings = [
        {**SAMPLE_LISTING, 'source_url': 'https://ex.com/1'},
        {**SAMPLE_LISTING, 'source_url': 'https://ex.com/2'},
        {**SAMPLE_LISTING, 'source_url': 'https://ex.com/1'},  # duplicate
    ]
    result = insert_listings(conn, listings)
    assert result == (2, 1)
    conn.close()


def test_inserted_row_has_scraped_at(tmp_path):
    conn = init_db(str(tmp_path / 'test.db'))
    insert_listings(conn, [SAMPLE_LISTING])
    row = conn.execute('SELECT scraped_at FROM listings').fetchone()
    assert row[0] is not None
    # Verify it parses as ISO-8601
    dt = datetime.strptime(row[0], '%Y-%m-%dT%H:%M:%SZ')
    assert dt.year >= 2026
    conn.close()


def test_insert_empty_list_returns_0_0(tmp_path):
    conn = init_db(str(tmp_path / 'test.db'))
    result = insert_listings(conn, [])
    assert result == (0, 0)
    conn.close()
