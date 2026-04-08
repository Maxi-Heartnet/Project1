---
date: 2026-04-07
topic: webcrawler-db-retrain
---

# Web Crawler, Database, and Retraining Pipeline

## Problem Frame

The model was trained on a one-time scrape of corotos.com.do (apartments only, ~30 pages). As
the DR housing market moves, predictions drift and there is no mechanism to collect fresh data or
retrain with it. The existing `scraper/scraper.py` is a working foundation but targets a single
site and writes only to a CSV that is overwritten each run.

This work adds: (1) a multi-site scraper covering up to 10 DR real estate portals (confirmed via
pre-implementation spike), (2) a SQLite database that accumulates listings across runs with
URL-based deduplication, and (3) a manual CLI retraining workflow that exports from the DB,
produces an updated `model.pkl`, and logs quality metrics for developer review before commit.

## Data Flow

```mermaid
TB
  S1[corotos.com.do] --> SCR[scraper/]
  S2[supercasas.com] --> SCR
  S3[miscasasrd.com] --> SCR
  S4[remaxrd.com] --> SCR
  S5[indominicana.com] --> SCR
  S6[apartamentosrd.com.do] --> SCR
  S7[tucasard.com] --> SCR
  S8[inmuebles.mercadolibre.com.do] --> SCR
  S9[plusval.com.do] --> SCR
  S10[casaspb.com] --> SCR
  SCR --> DB[(SQLite\nlistings.db)]
  DB --> EXP[export to\nlistings.csv]
  EXP --> TRAIN[ml/train.py]
  TRAIN --> PKL[model.pkl + metrics]
  PKL --> API[api.py]
```

## Requirements

**Scraper**
- R0. Before implementation begins, run a scrapability spike on all 10 sites: classify each as
  Phase 1 (confirmed scrapeable with requests+BeautifulSoup), Phase 2 (JS rendering required),
  or Blocked (auth wall / anti-bot). Planning proceeds only with Phase 1 sites confirmed.
- R1. The scraper targets Phase 1 sites from the spike. Aspirational list: corotos.com.do,
  supercasas.com, miscasasrd.com, remaxrd.com, indominicana.com, apartamentosrd.com.do,
  tucasard.com, inmuebles.mercadolibre.com.do, plusval.com.do, casaspb.com.
- R2. Each site has its own parser module at `scraper/sites/<site>.py` (e.g.
  `scraper/sites/corotos.py`). Each module exports a `scrape()` function that returns a list
  of dicts. All parsers extract five fields: `price` (USD integer), `sector`, `property_type`
  (`apartment` or `house`), `bedrooms`, `area_m2`. Non-USD listings (DOP/RD$) are skipped.
  Listings with any of the five fields missing are skipped and logged. The corotos.com.do
  parser logic (USD price detection, field extraction patterns) is the reference for new
  parsers; CSS selectors are site-specific and each site requires its own implementation.
- R3. `scraper/__main__.py` maintains a registry dict mapping site name to parser module.
  `python -m scraper` iterates the registry, calls each site's `scrape()`, writes results to
  the DB, and prints a per-site summary: "corotos: 147 new, 23 skipped (dedup)". Phase 2 /
  Blocked sites are noted in the registry but not called.
- R4. Sites that fail at runtime (network error, parse error, anti-bot block) are logged at
  error level with site name and exception. The CLI continues to remaining sites and exits
  with code 0. A final summary line shows: "Scraped N of M sites; K failed." Partial results
  from successful sites are written to the DB.

**Database**
- R5. A SQLite database at `data/listings.db` stores all scraped listings. Schema:
  `id` (INTEGER PRIMARY KEY), `price`, `sector`, `property_type`, `bedrooms`, `area_m2`,
  `source_url` (TEXT UNIQUE), `scraped_at` (TEXT, ISO-8601 UTC). The `source_url` column has
  a UNIQUE constraint to prevent duplicate inserts.
- R6. URL-based deduplication: before inserting, the scraper issues
  `INSERT OR IGNORE INTO listings ... WHERE source_url = ?`. Skipped duplicates are counted
  and included in the per-site summary (R3). `source_url` stores the full canonical URL
  including scheme and host (e.g., `https://corotos.com.do/anuncio/12345`).
- R7. `data/listings.db` must be added to `.gitignore` (verified missing; `data/listings.csv`
  is already gitignored).

**Retraining**
- R8. A CLI command (`python -m scraper export` or similar) exports cleaned rows from the DB
  to `data/listings.csv`, overwriting the file in place. Cleaned rows: non-null price and
  area_m2; price within `PRICE_MIN`/`PRICE_MAX` from `ml/prepare.py`. The export includes
  only the five training columns (`price`, `sector`, `property_type`, `bedrooms`, `area_m2`)
  — DB-internal columns (`id`, `source_url`, `scraped_at`) are excluded so `ml/train.py`
  receives the same schema as today. If no rows pass cleaning, export fails with a clear
  error message.
- R9. `ml/train.py` requires no changes — it reads `data/listings.csv` exactly as today.
- R10. After retraining, `ml/train.py` prints MAE and R² for the new model alongside any
  previously recorded values (if available) so the developer can compare before deciding
  whether to commit `ml/model.pkl`.
- R11. After the developer reviews metrics and decides to commit, they push `ml/model.pkl` to
  git and redeploy to EC2 via the existing pattern (git pull + supervisorctl restart).

## Success Criteria
- Spike classifies all 10 aspirational sites as Phase 1, Phase 2, or Blocked before planning
- Running the scraper CLI adds new rows to the DB and skips URLs already seen on subsequent runs
- The CLI prints a per-site summary and a final "N of M sites" line
- The DB can be exported to `data/listings.csv` and fed to `ml/train.py` without errors
- Retraining prints MAE and R² to stdout before the developer commits
- Retraining on the combined dataset produces a `model.pkl` that loads and serves correctly in
  `api.py` without any changes to the API
- A developer can run the full pipeline end-to-end: scrape → export → train → review metrics
  → commit → deploy

## Scope Boundaries
- No scrape scheduling or automation — manual CLI invocation only
- No API endpoint for triggering scrapes or retraining
- No DOP/RD$ listings — USD only (existing behavior preserved)
- No UI for browsing the DB or reviewing collected listings
- No changes to `ml/train.py`, `ml/prepare.py`, `api.py`, or `tests/`
- Sites classified as Phase 2 (JS rendering) or Blocked are deferred to a future iteration
- URL dedup accepted limitation: re-listed properties with new URLs will be treated as new
  listings (no content-hash dedup in this iteration)

## Key Decisions
- **Scrapability spike before implementation (R0)**: Prevents committing to sites that turn out
  to require headless browsers. Spike output becomes the definitive Phase 1 site list.
- **One file per site**: `scraper/sites/<site>.py` with a `scrape()` function. Registry dict in
  `scraper/__main__.py` dispatches to each. Easy to add a site without touching other files.
- **SQLite over PostgreSQL**: Zero-setup, file-based, no extra service on EC2. Appropriate for a
  solo project with no concurrent write requirements.
- **URL-based dedup with UNIQUE constraint**: Simple and reliable; INSERT OR IGNORE handles
  conflicts at the DB level without application-level SELECT queries.
- **Export-to-CSV bridge**: `ml/train.py` reads CSV today; writing from DB to CSV avoids
  touching the training pipeline entirely.
- **Manual retraining with metric logging**: Developer compares new MAE/R² to previous before
  committing. Prevents silent degradation from noisy scrape data.
- **Commit model.pkl pattern**: Consistent with the existing deploy workflow; no new
  infrastructure or secrets needed.

## Dependencies / Assumptions
- Scrapability spike (R0) determines the actual Phase 1 site list; the 10 aspirational sites
  in R1 may not all be achievable with requests+BeautifulSoup.
- `data/listings.db` is **not** in `.gitignore` and must be added (R7). `data/listings.csv`
  is already gitignored.
- `ml/train.py` is deterministic for a given CSV input (both RandomForest and KMeans use
  `random_state=42`); this must be preserved for reproducible retraining.
- Existing `scraper/scraper.py` provides the corotos parser as a reference implementation;
  it remains importable and is not rewritten (its `parse_listing()` and `parse_price_usd()`
  functions may be called or copied as the corotos site parser module).

## Outstanding Questions

### Resolve Before Planning
*(none)*

### Deferred to Planning
- [Affects R1, R2][Needs research] Spike: which of the 10 aspirational sites are scrapeable
  with requests+BeautifulSoup? What CSS selectors and pagination patterns do they use?
- [Affects R2, R4][Needs research] Which sites require JavaScript rendering? → Phase 2.
- [Affects R4][Technical] Should the existing `data/raw/` HTML cache directory be extended to
  all Phase 1 sites, kept corotos-only, or dropped in favor of live-only fetching?
- [Affects R7][Technical] Add `data/listings.db` to `.gitignore` before first scrape run.
- [Affects R10][Technical] How should "previously recorded values" be stored for metric
  comparison? (e.g., a `ml/metrics.json` file committed alongside `model.pkl`, or just
  printed from the most recent train run in the terminal.)

## Next Steps
→ `/ce:plan` for structured implementation planning
