import csv
import json
import sys
from io import StringIO
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

import scraper.__main__ as main_mod
from scraper.__main__ import cmd_export, cmd_scrape

SAMPLE_LISTING = {
    'price': 185000,
    'sector': 'Piantini',
    'property_type': 'apartment',
    'bedrooms': 3,
    'area_m2': 120.0,
    'source_url': 'https://corotos.com.do/anuncio/12345',
}


def _make_mock_site(listings=None, raises=None) -> ModuleType:
    m = MagicMock(spec=['scrape'])
    if raises:
        m.scrape.side_effect = raises
    else:
        m.scrape.return_value = listings or [SAMPLE_LISTING]
    return m


# ---------------------------------------------------------------------------
# cmd_scrape tests
# ---------------------------------------------------------------------------

def test_cmd_scrape_calls_each_registry_site(tmp_path):
    mock_site = _make_mock_site()
    registry = {'corotos': mock_site}
    with patch.object(main_mod, 'REGISTRY', registry):
        cmd_scrape(db_path=str(tmp_path / 'test.db'), status_path=str(tmp_path / 'status.json'))
    mock_site.scrape.assert_called_once()


def test_cmd_scrape_prints_per_site_summary(tmp_path, capsys):
    listing1 = {**SAMPLE_LISTING, 'source_url': 'https://ex.com/1'}
    listing2 = {**SAMPLE_LISTING, 'source_url': 'https://ex.com/2'}
    listing_dup = {**SAMPLE_LISTING, 'source_url': 'https://ex.com/1'}  # duplicate
    mock_site = _make_mock_site([listing1, listing2])
    registry = {'corotos': mock_site}

    with patch.object(main_mod, 'REGISTRY', registry):
        cmd_scrape(db_path=str(tmp_path / 'test.db'), status_path=str(tmp_path / 'status.json'))

    captured = capsys.readouterr()
    assert 'corotos:' in captured.out
    assert '2 new' in captured.out


def test_cmd_scrape_dedup_summary_counts(tmp_path, capsys):
    """First run inserts 2 new; second run skips both as duplicates."""
    listing1 = {**SAMPLE_LISTING, 'source_url': 'https://ex.com/1'}
    listing2 = {**SAMPLE_LISTING, 'source_url': 'https://ex.com/2'}
    mock_site = _make_mock_site([listing1, listing2])
    registry = {'corotos': mock_site}
    db_path = str(tmp_path / 'test.db')
    status_path = str(tmp_path / 'status.json')

    with patch.object(main_mod, 'REGISTRY', registry):
        cmd_scrape(db_path=db_path, status_path=status_path)
        capsys.readouterr()  # clear first run output
        cmd_scrape(db_path=db_path, status_path=status_path)

    captured = capsys.readouterr()
    assert '0 new' in captured.out
    assert '2 skipped' in captured.out


def test_cmd_scrape_continues_after_site_failure(tmp_path):
    import requests
    failing = _make_mock_site(raises=requests.RequestException('timeout'))
    succeeding = _make_mock_site([{**SAMPLE_LISTING, 'source_url': 'https://ex.com/ok'}])
    registry = {'bad_site': failing, 'good_site': succeeding}

    db_path = str(tmp_path / 'test.db')
    with patch.object(main_mod, 'REGISTRY', registry):
        cmd_scrape(db_path=db_path, status_path=str(tmp_path / 'status.json'))

    # Good site's listing should be in DB
    import sqlite3
    conn = sqlite3.connect(db_path)
    count = conn.execute('SELECT COUNT(*) FROM listings').fetchone()[0]
    conn.close()
    assert count == 1


def test_cmd_scrape_prints_final_summary(tmp_path, capsys):
    mock_site = _make_mock_site()
    registry = {'corotos': mock_site}
    with patch.object(main_mod, 'REGISTRY', registry):
        cmd_scrape(db_path=str(tmp_path / 'test.db'), status_path=str(tmp_path / 'status.json'))
    captured = capsys.readouterr()
    assert 'Scraped' in captured.out
    assert 'sites' in captured.out
    assert 'failed' in captured.out


def test_cmd_scrape_final_summary_on_partial_failure(tmp_path, capsys):
    import requests
    failing = _make_mock_site(raises=requests.RequestException('err'))
    succeeding = _make_mock_site()
    registry = {'bad': failing, 'good': succeeding}
    with patch.object(main_mod, 'REGISTRY', registry):
        cmd_scrape(db_path=str(tmp_path / 'test.db'), status_path=str(tmp_path / 'status.json'))
    captured = capsys.readouterr()
    assert '1 of 2' in captured.out
    assert '1 failed' in captured.out


def test_cmd_scrape_deduplication_across_runs(tmp_path):
    listing = {**SAMPLE_LISTING, 'source_url': 'https://ex.com/1'}
    mock_site = _make_mock_site([listing])
    registry = {'corotos': mock_site}
    db_path = str(tmp_path / 'test.db')
    status_path = str(tmp_path / 'status.json')

    with patch.object(main_mod, 'REGISTRY', registry):
        cmd_scrape(db_path=db_path, status_path=status_path)
        mock_site.scrape.return_value = [listing]
        cmd_scrape(db_path=db_path, status_path=status_path)

    import sqlite3
    conn = sqlite3.connect(db_path)
    count = conn.execute('SELECT COUNT(*) FROM listings').fetchone()[0]
    conn.close()
    assert count == 1


def test_cmd_scrape_writes_status_json(tmp_path):
    mock_site = _make_mock_site()
    registry = {'corotos': mock_site}
    status_path = str(tmp_path / 'status.json')
    with patch.object(main_mod, 'REGISTRY', registry):
        cmd_scrape(db_path=str(tmp_path / 'test.db'), status_path=status_path)
    with open(status_path) as f:
        status = json.load(f)
    assert status['sites_attempted'] == 1
    assert status['sites_succeeded'] == 1
    assert status['sites_failed'] == 0


# ---------------------------------------------------------------------------
# cmd_export tests
# ---------------------------------------------------------------------------

def _setup_db_with_listings(db_path, listings):
    """Helper to insert listings into a tmp DB for export tests."""
    from scraper.db import init_db, insert_listings
    conn = init_db(db_path)
    insert_listings(conn, listings)
    conn.close()


def test_cmd_export_writes_five_columns_only(tmp_path):
    db_path = str(tmp_path / 'test.db')
    csv_path = str(tmp_path / 'out.csv')
    _setup_db_with_listings(db_path, [SAMPLE_LISTING, {**SAMPLE_LISTING, 'source_url': 'https://ex.com/2'}])
    cmd_export(db_path=db_path, csv_path=csv_path, status_path=str(tmp_path / 'status.json'))
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ['price', 'sector', 'property_type', 'bedrooms', 'area_m2']


def test_cmd_export_excludes_source_url_and_id(tmp_path):
    db_path = str(tmp_path / 'test.db')
    csv_path = str(tmp_path / 'out.csv')
    _setup_db_with_listings(db_path, [SAMPLE_LISTING])
    cmd_export(db_path=db_path, csv_path=csv_path, status_path=str(tmp_path / 'status.json'))
    with open(csv_path) as f:
        content = f.read()
    assert 'source_url' not in content
    assert 'scraped_at' not in content


def test_cmd_export_applies_price_bounds(tmp_path):
    db_path = str(tmp_path / 'test.db')
    csv_path = str(tmp_path / 'out.csv')
    cheap = {**SAMPLE_LISTING, 'price': 1, 'source_url': 'https://ex.com/cheap'}
    valid = {**SAMPLE_LISTING, 'price': 200000, 'source_url': 'https://ex.com/valid'}
    _setup_db_with_listings(db_path, [cheap, valid])
    cmd_export(db_path=db_path, csv_path=csv_path, status_path=str(tmp_path / 'status.json'))
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert int(rows[0]['price']) == 200000


def test_cmd_export_fails_when_no_valid_rows(tmp_path):
    db_path = str(tmp_path / 'test.db')
    csv_path = str(tmp_path / 'out.csv')
    bad = {**SAMPLE_LISTING, 'price': 1, 'source_url': 'https://ex.com/bad'}
    _setup_db_with_listings(db_path, [bad])
    with pytest.raises(SystemExit) as exc_info:
        cmd_export(db_path=db_path, csv_path=csv_path, status_path=str(tmp_path / 'status.json'))
    assert exc_info.value.code != 0


def test_cmd_export_overwrites_existing_csv(tmp_path):
    db_path = str(tmp_path / 'test.db')
    csv_path = str(tmp_path / 'out.csv')
    # Pre-populate CSV with different data
    with open(csv_path, 'w') as f:
        f.write('old,data\n1,2\n')
    _setup_db_with_listings(db_path, [SAMPLE_LISTING])
    cmd_export(db_path=db_path, csv_path=csv_path, status_path=str(tmp_path / 'status.json'))
    with open(csv_path) as f:
        content = f.read()
    assert 'old' not in content
    assert 'price' in content


def test_cmd_export_warns_on_partial_scrape_failure(tmp_path, capsys):
    db_path = str(tmp_path / 'test.db')
    csv_path = str(tmp_path / 'out.csv')
    status_path = str(tmp_path / 'status.json')
    _setup_db_with_listings(db_path, [SAMPLE_LISTING])
    # Write status indicating 1 failure
    with open(status_path, 'w') as f:
        json.dump({'sites_failed': 1, 'failed_sites': ['bad_site']}, f)
    cmd_export(db_path=db_path, csv_path=csv_path, status_path=status_path)
    captured = capsys.readouterr()
    assert 'WARNING' in captured.err
    assert 'failure' in captured.err


def test_cmd_export_no_warning_when_no_failures(tmp_path, capsys):
    db_path = str(tmp_path / 'test.db')
    csv_path = str(tmp_path / 'out.csv')
    status_path = str(tmp_path / 'status.json')
    _setup_db_with_listings(db_path, [SAMPLE_LISTING])
    with open(status_path, 'w') as f:
        json.dump({'sites_failed': 0, 'failed_sites': []}, f)
    cmd_export(db_path=db_path, csv_path=csv_path, status_path=status_path)
    captured = capsys.readouterr()
    assert 'WARNING' not in captured.err
