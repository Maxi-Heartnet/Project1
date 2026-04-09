---
title: indominicana.com Area and Sector Silently Return None Due to get_text Separator and Comma-Leading Anchor
date: 2026-04-09
category: logic-errors
module: scraper/sites/indominicana.py
problem_type: logic_error
component: tooling
severity: high
symptoms:
  - area_m2 is None for all cards even though area is shown on the page
  - sector is None for all cards even though location is shown on the page
  - Scraper returns 0 listings; all cards skipped due to missing required fields
root_cause: logic_error
resolution_type: code_fix
tags:
  - web-scraping
  - beautifulsoup
  - get_text
  - regex
  - html-parsing
  - indominicana
---

# indominicana.com Area and Sector Silently Return None Due to get_text Separator and Comma-Leading Anchor

## Problem

The indominicana.com scraper returned 0 listings because two independent parsing bugs caused `area_m2` and `sector` to resolve to `None` for every card. No exception was raised — the scraper silently skipped all listings as "field-incomplete".

## Symptoms

- `indominicana: 0 new, 0 skipped (dedup)` in scrape output
- Debug logging shows repeated `missing area` and `missing sector` for every card
- Both fields are visually present on the live website
- No regex match error — the patterns simply don't match the extracted text

## What Didn't Work

- Assuming `get_text(separator=' ')` was safe for all HTML structures. The `separator` parameter inserts the separator string *between every tag boundary*, including around `<sup>` tags — so `m<sup>2</sup>` becomes "m 2" (with a space), not "m2".
- Calling `a_text.split(',')[0]` directly on anchor text. When the anchor text is `", Santo Domingo Este"` (a leading comma is present in the live HTML), `split(',')[0]` returns an empty string.

## Solution

**Fix 1 — Area extraction: remove `separator` from `get_text`**

```python
# Before — separator splits m<sup>2</sup> into "m 2"
card_text = card.get_text(separator=' ', strip=True)
# regex r'(\d+(?:[.,]\d+)?)\s*m[²2]' fails to match "m 2"

# After — no separator; m<sup>2</sup> becomes "m2"
card_text = card.get_text(strip=True)
```

**Fix 2 — Sector extraction: strip leading commas before splitting**

```python
# Before — fails for ", Santo Domingo Este" (split(',')[0] == '')
for a in card.find_all('a', href=True):
    a_text = a.get_text(strip=True)
    if a_text and ',' in a_text:
        sector = a_text.split(',')[0].strip()
        break

# After — filters empty parts after split
for a in card.find_all('a', href=True):
    a_text = a.get_text(strip=True)
    if not a_text:
        continue
    parts = [p.strip() for p in a_text.split(',') if p.strip()]
    if parts:
        sector = parts[0]
        break
```

## Why This Works

**Area:** BeautifulSoup's `get_text(separator=X)` inserts `X` at every tag boundary — including between a tag and its child inline elements like `<sup>`. With `separator=' '`, `m<sup>2</sup>` becomes "m" + " " + "2" = "m 2". The regex `m[²2]` requires "m" immediately followed by "2" or "²", so it never matches. Removing the separator lets the HTML "m<sup>2</sup>" collapse to "m2", which the regex handles correctly.

**Sector:** The location anchor on indominicana.com contains text like `", Santo Domingo Este"` — the comma precedes the location name, it does not follow it. `split(',')` on this text produces `['', ' Santo Domingo Este']`. Taking index 0 returns the empty string. Filtering out empty parts after split and taking the first remaining part correctly extracts "Santo Domingo Este".

## Prevention

- **Test with the actual anchor text format:** add a fixture that mirrors the live HTML pattern `, Location Name` (leading comma) and assert the sector is non-empty.
- **Log raw text for None fields at DEBUG level:**
  ```python
  if area_m2 is None:
      logger.debug('indominicana: area regex did not match. card_text repr: %r', card_text[:200])
  ```
  This makes silent parsing failures visible without needing a live network call.
- **Avoid `get_text(separator=...)` for regex matching** unless you specifically need the separator to separate sibling text nodes. When matching sub-element text (like "m²"), use `get_text(strip=True)` with no separator, or extract the specific element's text directly.

## Related Issues

- `docs/plans/spike-results.md` — original CSS selectors; location anchor complexity noted but parsing edge cases not documented
- `docs/solutions/runtime-errors/supercasas-stale-css-selectors-2026-04-09.md` — related scraper DOM issue from the same session
