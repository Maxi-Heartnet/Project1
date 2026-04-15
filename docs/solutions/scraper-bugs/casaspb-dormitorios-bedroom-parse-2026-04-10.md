---
title: casaspb.com _parse_bedrooms Regex Missed 'dormitorios' Label
date: 2026-04-10
category: scraper-bugs
module: scraper/sites/casaspb.py
problem_type: selector_drift
symptoms:
  - casaspb returns 0 new, 0 skipped (dedup) — silent zero output
  - No exception raised; scraper completes normally
  - All cards silently dropped by required-field validation
root_cause: EasyBroker platform changed bedroom label from abbreviated 'hab' to full Spanish 'dormitorios'; regex did not cover new vocabulary
resolution_type: regex_fix
tags:
  - casaspb
  - bedroom-parsing
  - regex
  - selector-drift
  - dormitorios
  - easybroker
---

# casaspb.com _parse_bedrooms Regex Missed 'dormitorios' Label

## Problem

The casaspb.com scraper returned 0 listings after the EasyBroker platform updated its listing card template to use the full Spanish word `dormitorios` for bedroom counts instead of the abbreviated form `hab`. Because `_parse_bedrooms()` did not match the new label, it returned `None` for every card — and bedrooms being a required field caused all cards to be silently discarded.

## Symptoms

- Post-scrape output: `casaspb: 0 new, 0 skipped (dedup)` while all other sites returned data normally
- No exceptions, no error logs — scraper ran to completion silently
- `0 skipped (dedup)` is the key signal: cards weren't even reaching deduplication, meaning they were filtered out at required-field validation

## What Didn't Work

No alternative approaches were needed. The `0 new, 0 skipped` signature immediately pointed to a required-field drop rather than a network or pagination issue. A live fetch and manual card inspection revealed `dormitorios` where `hab` was previously expected.

## Solution

In `scraper/sites/casaspb.py`, add `dormitorios` to the `_parse_bedrooms()` regex alternation:

```python
# Before — missed current EasyBroker vocabulary
def _parse_bedrooms(text):
    """Return bedroom count from text containing bed/hab/recámara indicator, or None."""
    m = re.search(r'(\d+)\s*(?:hab|bed|rec[aá]mara)', text, re.IGNORECASE)

# After
def _parse_bedrooms(text):
    """Return bedroom count from text containing bed/hab/dormitorio/recámara indicator, or None."""
    m = re.search(r'(\d+)\s*(?:hab|bed|dormitorio|rec[aá]mara)', text, re.IGNORECASE)
```

Result: 1,978 listings collected across ~149 pages.

**Also apply to `scraper/sites/miscasasrd.py`:** both sites run on the EasyBroker platform and contain identical copies of `_parse_bedrooms()`. A platform-level template change affects both scrapers simultaneously. The `miscasasrd.py` fix should be applied in the same commit.

## Why This Works

EasyBroker updated its listing card template to render bedroom counts as `"3 dormitorios"` instead of `"3 hab"`. The original regex alternation did not include `dormitorios`, so `re.search()` returned `None` on every card. Because bedrooms is validated as a required field, all cards failed and were dropped before insertion — producing a structurally successful scrape run with zero output.

Adding `dormitorios` to the alternation group restores matching against the current platform output. The `re.IGNORECASE` flag handles capitalisation; the existing `\s*` quantifier handles whitespace between digit and label.

## Prevention

1. **`0 new, 0 skipped` is an alert signal.** This combination means the scraper fetched pages and produced no records — not even candidates for deduplication. Any site showing this pattern on a run that previously produced data should be inspected for vocabulary or DOM changes before assuming the site had no new listings.

2. **Apply fixes to all EasyBroker sites simultaneously.** `casaspb.py` and `miscasasrd.py` share the same platform and duplicated `_parse_bedrooms()` implementation. When a platform-level label changes, update both scrapers in the same commit.

3. **Broaden the bedroom regex proactively** to cover the full known set of Spanish and English labels, reducing exposure to future cosmetic platform updates:

   ```python
   r'(\d+)\s*(?:hab(?:itaciones?)?|dormitorio[s]?|bed(?:rooms?)?|rec[aá]mara[s]?)'
   ```

   This covers: `hab`, `habitaciones`, `dormitorios`, `bed`, `bedrooms`, `recámara` — all observed variants across DR real estate sites.

4. **Update test fixtures when the platform vocabulary changes.** The existing `tests/test_sites_casaspb.py` fixtures use `"3 hab | 95 m²"` — update these to also include `"3 dormitorios"` variants so the test suite catches future regressions.

## Related

- `docs/solutions/runtime-errors/supercasas-stale-css-selectors-2026-04-09.md` — same failure mode (0 listings, silent drop) on supercasas.com; root cause was broader CSS selector drift, but `_parse_bedrooms()` was also updated there to handle `Habitaciones :` format
- `docs/plans/spike-results.md` — casaspb.com selectors last validated 2026-04-08
