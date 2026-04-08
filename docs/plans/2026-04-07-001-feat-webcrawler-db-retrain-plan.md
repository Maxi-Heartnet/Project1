---
date: 2026-04-07
type: feat
status: active
origin: docs/brainstorms/2026-04-07-webcrawler-db-retrain-requirements.md
deepened: 2026-04-07
---

# feat: Multi-site web scraper, SQLite accumulator, and retraining pipeline

## Problem Frame

The prediction model was trained on a one-time scrape of corotos.com.do (~30 pages). There is no
mechanism to collect fresh listings or retrain as the DR market moves. This plan adds:

1. A scrapability spike to classify 10 aspirational DR real estate sites
2. A SQLite database (`data/listings.db`) that accumulates listings across runs with URL-based deduplication
3. Per-site parser modules under `scraper/sites/` with a registry dispatcher in `scraper/__main__.py`
4. An export CLI (`python -m scraper export`) that writes `data/listings.csv` for `ml/train.py`
5. Metric comparison logging in `ml/train.py` (MAE/R² vs prior run) before the developer commits `model.pkl`

## Research Summary

**Technology:** Python 3.13 · requests + beautifulsoup4 (already in requirements.txt) · sqlite3 (stdlib, no new dep) · scikit-learn 1.6.1 (pinned — must stay pinned, see institutional constraint)

**What exists:**
- `scraper/scraper.py` — working corotos.com.do parser; `parse_listing()`, `fetch_page()`, `parse_price_usd()` are importable; `scrape() -> list[dict]` returns 5-field dicts; this is the reference interface
- `scraper/__init__.py` — empty; `scraper/__main__.py` and `scraper/sites/` do not exist yet
- `ml/train.py:train()` — prints MAE/R² inline, returns None; needs minimal addition for R10
- `data/listings.db` — absent from `.gitignore` (must be added)
- `data/listings.csv` — already gitignored
- Test files: `tests/test_scraper.py` imports from `scraper.scraper` directly — that module must remain unchanged

**Institutional constraint:** scikit-learn must stay pinned at 1.6.1 (documented in
`docs/solutions/runtime-errors/scikit-learn-pickle-version-mismatch-2026-04-05.md`). Update the pin
atomically with any retraining commit. Never allow a version drift between training and deployment.

**Sector casing:** `scraper.scraper:parse_listing()` stores sector as raw `get_text(strip=True)` with
no normalization. The existing `data/listings.csv` has mixed casing across rows. Commit `b3c1379`
explicitly removed `.title()` from the API to preserve encoder compatibility. Sectors are stored as-is
throughout the pipeline (scraper → DB → export → train). No normalization is added at any layer.
If casing normalization is ever needed, it should be applied uniformly at export time matching the
exact form the encoder was trained on.

## Scope Boundary Resolution

**R10 vs "No changes to ml/train.py":** The scope boundary was written to prevent rearchitecting the
training pipeline. R10 requires metric comparison; the minimal fix is a targeted addition to `ml/train.py`
— read `ml/metrics.json` before fitting, write after, print both. A `metrics_path=None` parameter is
added to `train()` (defaulting to the anchored `ml/metrics.json`) so existing tests can redirect the
write to `tmp_path` without touching `tests/`. All other aspects (CSV schema, artifact schema, entry
point, random states) remain unchanged.

## Implementation Units

### Unit 0: Scrapability spike — Phase 1 site classification (R0)

**Gate:** Do not begin Units 2, 3, or 5 until spike is complete. Unit 1 and Unit 6 can start immediately.

**Site classifications:**
- **Phase 1** — requests + BeautifulSoup sufficient; price and location extractable from static HTML
- **Phase 2** — JavaScript rendering required; deferred to a future iteration
- **Blocked** — auth wall, anti-bot 403/429, or CAPTCHA; deferred

**What to do:**
- For each of the 9 non-corotos sites, make a `requests.get()` with a browser User-Agent, parse with BeautifulSoup, and attempt to extract at least one price and one sector/location field
- Sites to test: supercasas.com, miscasasrd.com, remaxrd.com, indominicana.com, apartamentosrd.com.do, tucasard.com, inmuebles.mercadolibre.com.do, plusval.com.do, casaspb.com
- For every Phase 1 site: document the CSS selectors for listing cards, price, location, and the anchor href; document the pagination URL pattern
- Record results in `docs/plans/spike-results.md`

**Spike output determines R1:** Only Phase 1 sites get parser modules in Unit 5. Phase 2 / Blocked
sites are noted in the registry as comments but not implemented.

**Files touched:** `docs/plans/spike-results.md` (new)

---

### Unit 1: Infrastructure — `.gitignore` addition and SQLite schema (R5, R6, R7)

- [ ] Add `data/listings.db` to `.gitignore` (append after the `data/listings.csv` line)
- [ ] Create `scraper/db.py`:
  - `REQUIRED_KEYS = {'price', 'sector', 'property_type', 'bedrooms', 'area_m2', 'source_url'}` — the
    exact 6 keys every listing dict must have before insert
  - `init_db(db_path: str) -> sqlite3.Connection` — creates the `listings` table if not exists; returns
    open connection
  - `insert_listings(conn, listings: list[dict]) -> tuple[int, int]` — validates that each dict contains
    exactly `REQUIRED_KEYS` (raises `ValueError` with a clear message naming the missing/extra keys if
    not); inserts row-by-row with `cursor.execute('INSERT OR IGNORE ...', ...)`, checking `cursor.rowcount`
    per row to accumulate reliable `new_count` and `skipped_count`; stores sectors as-is (no `.title()`);
    sets `scraped_at` to current UTC ISO-8601 timestamp; returns `(new_count, skipped_count)`
  - Schema: `id INTEGER PRIMARY KEY`, `price INTEGER`, `sector TEXT`, `property_type TEXT`,
    `bedrooms INTEGER`, `area_m2 REAL`, `source_url TEXT UNIQUE`, `scraped_at TEXT` (ISO-8601 UTC)

**Pattern reference:** No existing SQLite code — use Python stdlib `sqlite3` directly; no ORM dependencies.

**Files:**
- `.gitignore` (modify)
- `scraper/db.py` (new)
- `tests/test_db.py` (new)

**Test scenarios for `tests/test_db.py`:**
- `test_init_db_creates_table` — `init_db(tmp_path/'test.db')` does not raise; `SELECT name FROM sqlite_master WHERE type='table' AND name='listings'` returns a row
- `test_insert_new_listing_returns_1_0` — insert one valid listing; returns `(1, 0)`
- `test_insert_duplicate_url_returns_0_1` — insert same listing twice; second call returns `(0, 1)`; DB row count stays 1
- `test_insert_invalid_keys_raises_value_error` — insert dict missing `source_url`; raises `ValueError`
- `test_insert_stores_sector_as_is` — insert listing with `sector='piantini'`; query DB; row has `sector='piantini'` (no title-casing)
- `test_insert_batch_mixed_new_and_duplicate` — insert 3 listings, 2 unique + 1 duplicate URL; returns `(2, 1)`
- `test_inserted_row_has_scraped_at` — insert one listing; query DB; `scraped_at` is non-null and parses as ISO-8601

---

### Unit 2: Corotos site module (R2)

- [ ] Create `scraper/sites/__init__.py` (empty)
- [ ] Create `scraper/sites/corotos.py`:
  - Imports `parse_listing`, `fetch_page`, `BASE_URL` from `scraper.scraper` (unchanged)
  - Defines `LISTING_URL_SELECTOR` — the anchor tag selector to extract the canonical listing URL from
    each card; exact selector confirmed against `data/raw/page_1.html` during implementation (e.g.,
    `div.listing-item a[href]`); stores the full `https://corotos.com.do{href}` if href is relative
  - `scrape(max_pages=30, use_cache=True) -> list[dict]` — **reimplements the pagination loop** (does
    NOT call `scraper.scraper.scrape()`); iterates pages via `fetch_page()`, selects `div.listing-item`
    cards, for each card: calls `parse_listing(card)` for the 5 training fields, extracts `source_url`
    from the card's anchor; skips cards where `parse_listing()` returns any None field or where anchor
    extraction fails; returns list of 6-field dicts (`price, sector, property_type, bedrooms, area_m2, source_url`)
  - Non-USD listings are skipped by `parse_listing()` (returns `price=None`); field-incomplete cards
    are skipped and counted; the skip tally is logged at DEBUG level

**`scraper/scraper.py` must not change.** All tests in `tests/test_scraper.py` continue to pass.

**Files:**
- `scraper/sites/__init__.py` (new)
- `scraper/sites/corotos.py` (new)
- `tests/test_sites_corotos.py` (new)

**Test scenarios for `tests/test_sites_corotos.py`:**
- `test_scrape_returns_list` — call `scrape(use_cache=True)` with real cached HTML in `data/raw/`; result is a list
- `test_scrape_each_listing_has_required_keys` — every dict has all 6 keys: `price, sector, property_type, bedrooms, area_m2, source_url`
- `test_scrape_source_url_is_absolute` — all `source_url` values start with `https://`
- `test_scrape_skips_non_usd_listings` — mock `fetch_page` to return HTML with one RD$ listing; result is empty list
- `test_scrape_skips_listing_with_missing_field` — mock a card whose `parse_listing()` returns `price=None`; result excludes that card
- `test_scrape_empty_page_stops_pagination` — mock `fetch_page` to return HTML with no `div.listing-item` on page 2; scrape stops at page 1

---

### Unit 3: `scraper/__main__.py` — registry and scrape command (R3, R4)

- [ ] Create `scraper/__main__.py` with:
  - Registry dict at module top: `REGISTRY: dict[str, ModuleType]` — hardcoded explicit imports, one
    `import scraper.sites.<name> as <name>` per Phase 1 site; non-Phase-1 sites noted as comments with
    reason; adding a new site from Unit 5 requires manually adding its import here
  - `cmd_scrape(db_path)` — opens DB via `db.init_db(db_path)`; iterates registry; per site: calls
    `site.scrape()`, calls `db.insert_listings(conn, listings)`, prints per-site summary
    `"corotos: 147 new, 23 skipped (dedup)"`; catches all exceptions per-site at `logging.error` level
    with site name and exception; continues to remaining sites regardless; tracks failed site names;
    prints final summary `"Scraped N of M sites; K failed."`; exits 0
  - `if __name__ == '__main__'` block: parse `sys.argv[1]` for subcommand (`export` or default scrape);
    delegate to `cmd_scrape` / `cmd_export`

**Files:**
- `scraper/__main__.py` (new)
- `tests/test_scraper_main.py` (new)

**Test scenarios for `tests/test_scraper_main.py`:**
- `test_cmd_scrape_calls_each_registry_site` — patch registry mocks; call `cmd_scrape(tmp_db)`; each mock's `scrape()` called once
- `test_cmd_scrape_prints_per_site_summary` — mock one site returning 3 listings with 1 duplicate; stdout contains `"2 new, 1 skipped"`
- `test_cmd_scrape_continues_after_site_failure` — mock first site raises `requests.RequestException`; mock second returns 1 listing; `cmd_scrape` does not raise; second site's listing is in DB
- `test_cmd_scrape_exits_0_on_partial_failure` — invoke `cmd_scrape` with one failing site; no exception raised; exit code 0
- `test_cmd_scrape_prints_final_summary` — stdout contains line matching `"Scraped N of M sites; K failed."`
- `test_cmd_scrape_deduplication_across_runs` — run `cmd_scrape` twice with same mock data; DB row count equals unique URL count

---

### Unit 4: Export CLI (R8)

Implemented as `cmd_export` inside `scraper/__main__.py`.

- [ ] `cmd_export(db_path, csv_path)`:
  - Opens `db_path` with `sqlite3`
  - Queries: `SELECT price, sector, property_type, bedrooms, area_m2 FROM listings WHERE price IS NOT NULL AND area_m2 IS NOT NULL`
  - Applies price bounds: `PRICE_MIN = 10_000` and `PRICE_MAX = 5_000_000` defined as constants at
    module top in `scraper/__main__.py` with comment `# matches ml/prepare.py PRICE_MIN / PRICE_MAX`
    (no cross-package import from `ml.prepare`)
  - Applies sector normalization at export time: strip whitespace only (no `.title()`), matching the
    form the encoder was trained on
  - If no rows pass cleaning: print error to stderr, `sys.exit(1)`
  - Writes to `csv_path` using `csv.DictWriter` with `fieldnames=['price','sector','property_type','bedrooms','area_m2']` — 5 columns only
  - Overwrites `csv_path` in place
  - Prints: `"Exported N rows to {csv_path}"`
  - **Partial-failure warning:** if the most recent scrape had any failed sites (stored as a sidecar
    count in `data/scrape_status.json` written by `cmd_scrape`), print a prominent warning before
    writing: `"WARNING: last scrape had K site failure(s) — export may be incomplete. Review scrape logs before retraining."`

**Files:** `scraper/__main__.py` (Unit 3 file, continued)

**Test scenarios (add to `tests/test_scraper_main.py`):**
- `test_cmd_export_writes_five_columns_only` — insert 2 rows into tmp DB; call `cmd_export`; CSV header is exactly `['price','sector','property_type','bedrooms','area_m2']`
- `test_cmd_export_excludes_source_url_and_id` — CSV must not contain `source_url`, `id`, or `scraped_at`
- `test_cmd_export_applies_price_bounds` — insert `price=1` row and `price=200000` row; export; CSV contains only the valid row
- `test_cmd_export_fails_when_no_valid_rows` — insert only out-of-range rows; `cmd_export` raises `SystemExit` with non-zero code
- `test_cmd_export_overwrites_existing_csv` — pre-populate CSV with different data; run export; CSV contains only DB rows
- `test_cmd_export_warns_on_partial_scrape_failure` — write `data/scrape_status.json` indicating 1 failure; call `cmd_export`; stdout/stderr contains `"WARNING"` and `"failure"`

**`data/scrape_status.json` spec:** Written by `cmd_scrape` at the end of each run:
`{"sites_attempted": N, "sites_succeeded": N, "sites_failed": K, "failed_sites": ["name1", ...], "run_at": "<ISO-8601>"}`.
Not committed to git (add to `.gitignore` alongside `data/listings.db`).

---

### Unit 5: Additional Phase 1 site modules (R1, R2)

**Depends on Unit 0 (spike results).** Implement up to 5 Phase 1 sites beyond corotos in this
iteration. Additional sites beyond 5 are deferred to a future iteration.

For each Phase 1 site `<name>`:
- [ ] Create `scraper/sites/<name>.py` exporting `scrape() -> list[dict]` with 6 fields (5 training + `source_url`)
- [ ] Add `import scraper.sites.<name> as <name>` to the registry in `scraper/__main__.py`
- Reimplements its own pagination loop (same pattern as Unit 2's corotos.py)
- Site-specific CSS selectors defined as constants at module top (from spike-results.md)
- Non-USD listings skipped and counted; field-incomplete listings skipped and logged at DEBUG level
- `source_url` constructed as absolute `https://` URL from the card anchor

**Pattern reference:** `scraper/sites/corotos.py` (Unit 2).

**Files per site:**
- `scraper/sites/<name>.py` (new)
- `tests/test_sites_<name>.py` (new)

**Minimum test scenarios per site:**
- `test_scrape_returns_list`
- `test_scrape_each_listing_has_required_keys` — all 6 keys
- `test_scrape_source_url_is_absolute`
- `test_scrape_skips_non_usd_listings`
- `test_scrape_skips_listing_with_missing_field`

---

### Unit 6: Metric comparison logging in `ml/train.py` (R10)

Minimal targeted addition — function signature gains one optional parameter; all other aspects unchanged.

- [ ] Ensure `ml/metrics.json` is NOT in `.gitignore` — commit it alongside `model.pkl`
- [ ] Modify `ml/train.py`:
  - Add `metrics_path=None` to `train()`'s signature; when `None`, default to
    `os.path.join(os.path.dirname(__file__), 'metrics.json')` (anchored to `ml/` regardless of cwd)
  - At top of `train()`, before fitting: if `metrics_path` file exists, load it and print
    `"Previous: MAE $X, R² Y.YYY"`
  - After computing MAE/R², preserve existing print format; then write
    `{"mae": float, "r2": float, "trained_at": "<ISO-8601 UTC>"}` to `metrics_path`
  - No other changes: no new top-level imports beyond `json` and `os`, no CSV schema changes,
    no artifact schema changes, no random_state changes

**Files:**
- `ml/train.py` (minimal modification — signature + 5-line addition inside `train()`)
- `ml/metrics.json` — produced at runtime; committed after first retrain

**Test scenarios:** None added to `tests/` (scope boundary). Manual verification: run `python ml/train.py` twice; second run prints both "Previous:" and current metrics. Existing `test_train.py` tests pass `metrics_path=str(tmp_path/'metrics.json')` to redirect writes — **this is the one change allowed to `tests/test_train.py`**: add `metrics_path=str(tmp_path/'metrics.json')` to all existing `train()` calls in that file so tests don't write to the real `ml/metrics.json`.

---

## System-Wide Impact

| Area | Impact |
|---|---|
| `.gitignore` | Add `data/listings.db`, `data/scrape_status.json` |
| `docs/plans/` | New: `spike-results.md` (Unit 0 spike output) |
| `scraper/` package | New: `__main__.py`, `db.py`, `sites/__init__.py`, `sites/<site>.py` per Phase 1 site |
| `ml/train.py` | Minimal addition: `metrics_path` param + metrics.json read/write |
| `ml/metrics.json` | New committed artifact — written on first retrain |
| `data/listings.csv` | Now generated by export CLI (same schema) |
| `data/scrape_status.json` | New runtime file — tracks last scrape failure count; gitignored |
| `api.py` | No change |
| `tests/test_train.py` | Minimal change: add `metrics_path=str(tmp_path/'metrics.json')` to existing `train()` calls |
| `tests/` (other) | New test files only |
| EC2 deploy | No new Terraform changes; git pull + supervisorctl restart |

## Sequencing and Dependencies

```
Unit 0 (spike) ─────────────────────────────────────────────────┐
    │                                                            │
    ├─── Unit 1 (db.py + .gitignore)   ← parallel with Unit 0  │
    │        │                                                   │
    ├─── Unit 2 (sites/corotos.py)     ← after Unit 0          │
    │        │                                                   │
    ├─── Unit 3+4 (__main__.py)        ← after Units 1, 2      │
    │                                                            │
    └─── Unit 5 (≤5 Phase 1 sites)    ← after Unit 0 ──────────┘

Unit 6 (train.py metrics)  ← independent; can run at any point
```

Unit 1 can begin immediately — DB schema is fixed regardless of site count.
Unit 6 can be implemented at any time; no dependencies on other units.

## Key Decisions

| Decision | Rationale | Origin |
|---|---|---|
| `scraper/scraper.py` unchanged | Tests import it directly; breaking it breaks off-limits test file | Research finding |
| Sector stored as-is (no `.title()`) | Commit b3c1379 removed title-casing to preserve encoder compatibility; cannot regress | Adversarial review finding |
| Sector normalization only at export time | Strip whitespace at `cmd_export`; no other normalization | Review resolution |
| `PRICE_MIN`/`PRICE_MAX` duplicated in `scraper/__main__.py` | Avoids cross-package import; values are stable constants | Feasibility finding |
| `metrics_path` param added to `train()` | Allows existing tests to redirect write to `tmp_path` without touching test logic | Review resolution |
| `data/scrape_status.json` sidecar | Cheap way for `cmd_export` to warn about degraded data without coupling the two commands | Review resolution |
| SQLite stdlib only | No new pip dependency; appropriate for single-user CLI workload | (see origin: R key decisions) |
| `ml/metrics.json` committed, not gitignored | Prior metrics must survive fresh clone for comparison | Deferred Q from origin doc |
| One file per site | Easy to add/remove sites without touching other modules | (see origin: R key decisions) |
| Exit 0 on site failure | Partial results better than none; developer reviews per-site log | (see origin: R4) |
| Unit 5 capped at 5 additional sites | Bounds scope; remaining Phase 1 sites deferred to future iteration | Review resolution |
| `corotos.py` reimplements pagination loop | `scraper.scraper.scrape()` has no URL param and can't inject `source_url`; own loop required | Feasibility finding |

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Spike finds <3 Phase 1 sites | Medium | Corotos confirmed; even 1 additional site is a win; Phase 2 deferred not lost |
| Site DOM changes break parser | High (over months) | Per-site isolation; broken parser fails only that site; `cmd_scrape` continues |
| `source_url` varies between scraper versions | Medium | Each site module owns its URL construction; document expected URL form in module constants |
| ML encoder unseen sectors from new sites | Low | `handle_unknown='ignore'` already set; new sectors silently treated as unknown at predict time |
| scikit-learn version drift on retrain | Low | Pin at 1.6.1; institutional constraint documented in `docs/solutions/` |
| Stale `ml/metrics.json` after discarded retrain | Medium | See Execution Notes: `git checkout ml/metrics.json` after discarding a run |

## Deferred Questions (to implementation)

- **[Affects Unit 2]** Exact CSS selector for anchor tag URL in corotos listing cards — check `data/raw/page_1.html` first; confirm against live DOM
- **[Affects Unit 5]** Exact CSS selectors, pagination patterns, and request headers for each Phase 1 site — discovered during Unit 0 spike
- **[Affects Unit 0]** Whether any sites require custom `User-Agent` or `Referer` headers beyond `Mozilla/5.0`
- **[Affects Unit 5]** Pagination by offset vs. page number vs. cursor — each site module handles independently

## Execution Notes

- Run `python -m pytest tests/test_scraper.py` after Unit 2 — regression gate; all existing corotos tests must pass
- After Unit 4, run end-to-end smoke: `python -m scraper && python -m scraper export && python ml/train.py`
- After Unit 6, run `python ml/train.py` twice: second run must print "Previous:" line alongside current metrics
- Developer commits `ml/model.pkl` and `ml/metrics.json` together after reviewing metric comparison
- **If a retrain is discarded** (worse metrics, decision not to commit): run `git checkout ml/metrics.json` to restore the committed baseline before the next retraining run

## Success Criteria (from origin)

- Spike classifies all 10 aspirational sites before any site parser is written
- Running the scraper CLI adds rows to DB and skips seen URLs on subsequent runs
- CLI prints per-site summary and `"Scraped N of M sites; K failed."` final line
- DB can be exported to `data/listings.csv` and fed to `ml/train.py` without errors
- Retraining prints MAE and R² to stdout before developer commits
- Retraining produces a `model.pkl` that loads and serves correctly in `api.py` without any API changes
- Developer can run end-to-end: scrape → export → train → review metrics → commit → deploy
