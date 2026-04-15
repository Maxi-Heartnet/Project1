---
title: Windows CSV Write Fails with UnicodeEncodeError on Scraped Data Containing Emoji
date: 2026-04-09
category: runtime-errors
module: scraper/__main__.py
problem_type: runtime_error
component: tooling
severity: high
symptoms:
  - UnicodeEncodeError on Windows when running python -m scraper export
  - charmap codec cannot encode characters like U+1F4CD (📍 location pin)
  - Error only occurs on Windows; identical code works on Linux/macOS
root_cause: config_error
resolution_type: code_fix
tags:
  - unicode
  - windows
  - encoding
  - csv
  - file-handling
  - platform-specific
---

# Windows CSV Write Fails with UnicodeEncodeError on Scraped Data Containing Emoji

## Problem

`python -m scraper export` crashes on Windows when scraped listings contain Unicode characters (e.g., emoji like 📍). The CSV writer uses Python's default system encoding (cp1252 on Windows), which cannot represent characters outside the Windows-1252 code page.

## Symptoms

```
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f4cd'
in position 44: character maps to <undefined>
```

- Crash occurs in `scraper/__main__.py` at the `writer.writerows(clean)` call
- Export succeeds on Linux and macOS (both default to utf-8)
- Crash only appears after a *live* scrape — cached test data doesn't trigger it because test fixtures don't contain emoji

## What Didn't Work

- Opening the file without an `encoding=` argument. Python's `open()` uses `locale.getpreferredencoding(False)` as the default, which is cp1252 on Windows.
- Stripping Unicode characters before writing. This would silently drop location data.

## Solution

Add `encoding='utf-8'` to the `open()` call in `cmd_export`:

```python
# Before — uses system default encoding (cp1252 on Windows)
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(clean)

# After — explicit utf-8, works on all platforms
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(clean)
```

## Why This Works

Python's `open()` defaults to the locale encoding when no `encoding` is specified. On Windows this is typically cp1252 (Windows-1252), a single-byte encoding that covers Western European characters but cannot represent emoji or most non-Latin scripts. On Linux and macOS the default is utf-8, which covers the full Unicode range — this is why the bug is Windows-only. Explicitly specifying `encoding='utf-8'` ensures consistent behavior across platforms and aligns with how downstream consumers (pandas `read_csv`, the ML training pipeline) read the file.

## Prevention

- **Always specify `encoding='utf-8'` on every `open()` call** that reads or writes text files, unless a specific non-utf-8 encoding is required for compatibility with an external system.
- **Add a cross-platform test for the CSV write path** that includes a Unicode character in a fixture:
  ```python
  def test_cmd_export_handles_unicode_sector(tmp_path):
      # Insert listing with emoji in sector name (mimics live site data)
      conn = db.init_db(str(tmp_path / 'test.db'))
      db.insert_listings(conn, [{
          'price': 200000, 'sector': '📍 Piantini', 'property_type': 'apartment',
          'bedrooms': 3, 'area_m2': 120.0,
          'source_url': 'https://example.com/1'
      }])
      conn.close()
      csv_path = str(tmp_path / 'out.csv')
      cmd_export(db_path=str(tmp_path / 'test.db'), csv_path=csv_path,
                 status_path=str(tmp_path / 'status.json'))
      rows = list(csv.DictReader(open(csv_path, encoding='utf-8')))
      assert rows[0]['sector'] == '📍 Piantini'
  ```
- **Apply the same rule to all other file opens** in the scraper pipeline — log files, status JSON, etc. — to prevent the same issue from surfacing elsewhere.

## Related Issues

- `docs/solutions/runtime-errors/scikit-learn-pickle-version-mismatch-2026-04-05.md` — another platform/environment mismatch in the same pipeline
