---
title: supercasas.com CSS Selectors Stale After DOM Restructure
date: 2026-04-09
category: runtime-errors
module: scraper/sites/supercasas.py
problem_type: runtime_error
component: tooling
severity: high
symptoms:
  - Scraper returns 0 listings with no error raised
  - CSS selector li.normal matches only navbar items, not listing cards
  - Price and sector fields silently absent from all results
root_cause: logic_error
resolution_type: code_fix
tags:
  - web-scraping
  - css-selectors
  - dom-change
  - supercasas
  - beautifulsoup
---

# supercasas.com CSS Selectors Stale After DOM Restructure

## Problem

The supercasas.com scraper returned 0 listings after the site restructured its DOM. The selectors written during the initial spike were no longer valid, causing silent data loss — no error was raised, the scraper simply produced an empty result set.

## Symptoms

- `supercasas: 0 new, 0 skipped (dedup)` in scrape output (was previously hundreds of listings)
- No exception raised; scraper exits normally with 0 results
- `li.normal` matched 7 elements — all navbar items, not listing cards
- `.title3` selector returned `None` for every card inspected

## What Didn't Work

- Trusting the original spike selectors without re-validating against the live DOM after several months. The site had restructured without any breaking error signal.

## Solution

Update constants in `scraper/sites/supercasas.py` to match the current DOM:

```python
# Before — original spike selectors (no longer valid)
CARD_SELECTOR = 'li.normal'
PRICE_SELECTOR = '.title3'
LOCATION_SELECTOR = '.title2'

# After — current DOM (2026-04)
CARD_SELECTOR = '#bigsearch-results-inner-results li.special'
PRICE_SELECTOR = '.title2'   # now "Venta: US$ 330,000"
LOCATION_SELECTOR = '.title1'  # now holds sector/location
```

Update `_parse_bedrooms` to handle the new label format with fallback:

```python
def _parse_bedrooms(text):
    # New DOM format: "Habitaciones : 3"
    m = re.search(r'Habitaciones\s*:\s*(\d+)', text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # Fallback: old "3 hab" format
    m = re.search(r'(\d+)\s*hab', text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None
```

Update `_parse_area` to handle the new unit ("Mt2" instead of "m²"):

```python
def _parse_area(text):
    # Supercasas now uses "Mt2" e.g. "Construcción : 239 Mt2"
    m = re.search(r'(\d+(?:[.,]\d+)?)\s*[Mm]t?[2²]', text)
    if m:
        return float(re.sub(r',', '.', m.group(1)))
    return None
```

**Diagnosis method:** fetch the live page, inspect with BeautifulSoup:

```python
import requests
from bs4 import BeautifulSoup

resp = requests.get('https://www.supercasas.com/buscar/?Tipo=2&PagingPageSkip=0',
                    headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
soup = BeautifulSoup(resp.text, 'html.parser')

# Walk up from a known price element to find card container
sample = soup.select_one('div.title2')
el = sample
for _ in range(6):
    el = el.parent
    print(f'<{el.name} class={el.get("class")} id={el.get("id","")}>  len={len(el.get_text())}')
```

## Why This Works

Websites periodically restructure their DOM for design or performance reasons. supercasas.com moved listing cards from `li.normal` → `li.special` scoped inside `#bigsearch-results-inner-results`, swapped the semantic meaning of `.title1`/`.title2`, changed the bedrooms label from abbreviated Spanish (`hab`) to the full form (`Habitaciones :`), and changed the area unit from the Unicode superscript (`m²`) to a text abbreviation (`Mt2`). Updating selectors and regexes to match the live DOM restores full extraction.

## Prevention

- **Selector health check in CI or cron:** After each scrape run, log the count of cards matched per selector. Alert when a site drops to 0 cards (`"supercasas: 0 new, 0 skipped"` on a run that should yield results indicates selector rot, not just dedup).
- **Test fixture updates:** When a site's DOM changes, update the HTML fixture in `tests/test_sites_supercasas.py` to match the new structure before updating the selectors. The test failing first confirms you found the right change.
- **Selector comments:** Document the target element and its purpose alongside each selector constant, making future DOM audits faster:
  ```python
  # li.special inside #bigsearch-results-inner-results — each listing card (2026-04)
  CARD_SELECTOR = '#bigsearch-results-inner-results li.special'
  ```
- **Spike results doc:** When selectors are confirmed against the live DOM, update `docs/plans/spike-results.md` with the date of last validation.

## Related Issues

- `docs/plans/spike-results.md` — original CSS selectors from the scrapability spike (may now be stale for other sites)
- `docs/plans/2026-04-07-001-feat-webcrawler-db-retrain-plan.md` — risk: "Site DOM changes break parser | High (over months)"
