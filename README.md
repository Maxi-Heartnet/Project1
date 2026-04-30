# Santo Domingo House Price Estimator

**Video demo:** [PLACEHOLDER — add YouTube URL before submitting]

**Author:** Misael | GitHub: Maxi-Heartnet | Santo Domingo, Dominican Republic

---

## What Is This?

This is a full-stack machine learning web application that predicts real estate prices in Santo
Domingo, Dominican Republic. You enter a property's sector, type (apartment or house), number of
bedrooms, and area in square meters — and the app returns an estimated price range in USD along
with a market tier (Budget, Mid-Range, or Luxury) based on how the property compares to similar
listings in that sector.

The problem this solves is real: the Dominican Republic has no centralized MLS or public price
index, so buyers and sellers have almost no reliable data to anchor negotiations. This app
aggregates scraped listings from six real estate portals, trains a machine learning model on that
data, and exposes the predictions through a clean web interface with an interactive Google Maps
panel showing where each sector is located.

## How It Works

The application is built in three layers: a data pipeline, a trained model, and a web API with a
frontend.

**Data pipeline** — A multi-site web scraper (`scraper/`) visits six Dominican Republic real
estate portals and extracts listings with five fields: price (USD), sector, property type,
bedrooms, and area in m². Results are stored in a SQLite database (`data/listings.db`) with
URL-based deduplication so repeated scrape runs don't create duplicate rows. A CLI export command
generates `data/listings.csv` for training.

**Machine learning model** — `ml/train.py` loads the CSV, cleans and preprocesses the data
(one-hot encoding for categorical features, outlier filtering), and trains two models: a Random
Forest regressor for price prediction and a KMeans clustering model that classifies listings into
Budget, Mid-Range, or Luxury tiers by their price and size characteristics. The trained artifact
is serialized to `ml/model.pkl` with joblib.

**Web application** — A FastAPI server (`api.py`) loads the model at startup and exposes a
`/predict` REST endpoint. The frontend (`static/main.js`) submits the form via `fetch()`, formats
the response, and displays the price range and market tier. A Google Maps panel (`static/map.js`)
shows a pin for the selected sector and updates bidirectionally with the form — selecting a sector
pans the map, and clicking a pin fills the sector input.

## File and Directory Descriptions

| Path | Description |
|------|-------------|
| `api.py` | FastAPI application. Loads the model at startup, serves the HTML page with injected sector data, and handles `/predict` requests. |
| `ml/train.py` | Training script. Reads `data/listings.csv`, trains RandomForest + KMeans, saves `ml/model.pkl`, and prints MAE and R² metrics for review. |
| `ml/prepare.py` | Data preparation utilities shared by training and the API. Defines feature names, cleans the dataset, and builds the sklearn ColumnTransformer preprocessor. |
| `ml/model.pkl` | Serialized model artifact containing the regressor, encoder, clusterer, and supporting lookup tables. Not committed to git — generated locally. |
| `chatbot/chat.py` | CLI chatbot interface and shared prediction logic. `predict_range()` and `predict_tier()` are used by both the chatbot and the API endpoint. |
| `scraper/__main__.py` | Scraper CLI entry point. Iterates a registry of site parsers, calls each `scrape()` function, writes results to the DB, and prints a per-site summary. |
| `scraper/db.py` | SQLite database layer. Handles schema creation and `INSERT OR IGNORE` upserts for deduplication. |
| `scraper/sites/` | One module per real estate portal (`corotos.py`, `supercasas.py`, `miscasasrd.py`, `indominicana.py`, `mercadolibre.py`, `plusval.py`). Each exports a `scrape()` function that returns a list of listing dicts. |
| `static/main.js` | Frontend form logic. Handles validation, submission via `fetch()`, result rendering, and the "Fill randomly" and "Clear" buttons. |
| `static/map.js` | Google Maps panel. Creates `AdvancedMarkerElement` pins for each sector with known coordinates, handles bidirectional sync with the sector input, and shows a fallback if the Maps SDK fails to load. |
| `static/style.css` | Application stylesheet. Responsive layout using CSS Grid, no external framework. |
| `static/sector_coords.json` | Pre-seeded lat/lng coordinates for 62 of 72 sectors (86% coverage), used to place map pins. |
| `scripts/generate_sector_coords.py` | One-time geocoding script that uses the Google Geocoding API to generate `sector_coords.json`. Not part of the runtime application. |
| `terraform/` | Infrastructure-as-code for AWS EC2 deployment. Provisions instance, security group, Elastic IP, and IAM key. Passes Maps API credentials via `supervisor.conf.tpl`. |
| `deploy/supervisor.conf.tpl` | Supervisor process config template. Rendered by Terraform to inject environment variables at deploy time. |
| `tests/` | pytest test suite covering the API endpoints, ML pipeline, scraper database, site parsers, and Playwright browser tests for map interactions. |
| `data/listings.csv` | Training data CSV. Generated from the database; gitignored. |
| `docs/` | Brainstorm and planning documents created during development. |

## Design Decisions

**Why Random Forest for price prediction?** Random forests handle the mixed categorical/numeric
feature space well without requiring manual feature engineering, tolerate skewed distributions
(real estate prices have long tails), and naturally produce per-tree predictions that I used to
compute a price range: the 10th and 90th percentile of individual tree predictions. This gives
users a realistic band rather than a false point estimate. An alternative would have been gradient
boosting (XGBoost/LightGBM), which often achieves lower MAE but is less interpretable and harder
to explain to a non-technical audience.

**Why KMeans for market tiers?** Rather than hardcoding price bands, KMeans discovers clusters
in the price/size/bedrooms space from the data itself. The three clusters are then labeled
Budget, Mid-Range, and Luxury by sorting their mean prices. This means the tier boundaries adapt
automatically when the model is retrained on newer data, rather than requiring manual threshold
updates.

**Why SQLite instead of a hosted database?** The scraper runs infrequently (manually triggered,
not continuously), and the data volume is modest (~10k rows). SQLite requires zero infrastructure,
runs directly on the EC2 instance, and uses a UNIQUE constraint on `source_url` for deduplication
rather than application-level logic. PostgreSQL or another hosted database would add operational
complexity with no meaningful benefit at this scale.

**Why FastAPI instead of Flask or Django?** FastAPI generates OpenAPI documentation
automatically, validates request bodies using Python type hints and Pydantic models, and uses
ASGI for better async performance. For a JSON API with one meaningful endpoint, it is the
simplest framework that does the right things without ceremony.

**Why Terraform for deployment?** All infrastructure is defined in code: the EC2 instance type,
security group rules, key pair, Elastic IP, and supervisor configuration are reproducible and
version-controlled. Rebuilding the server from scratch takes one `terraform apply` with no manual
steps. This also made it straightforward to inject the Google Maps API key as a sensitive Terraform
variable rather than storing it in the repository or environment files.

## AI Tool Disclosure

This project was developed with the assistance of **Claude Code** (Anthropic), an AI coding
assistant. Claude Code contributed to code generation, debugging, test writing, infrastructure
configuration, and documentation. The design decisions, product direction, data sourcing, and
overall architecture were determined by the author. All AI-generated content was reviewed,
modified, and verified by the author before inclusion. Use of AI tools is permitted under CS50's
academic honesty policy; this disclosure satisfies the citation requirement.

## Developer Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# One-time: install Playwright browser binaries (for Playwright tests)
playwright install chromium
```

```bash
# Scrape all sites into data/listings.db
python -m scraper

# Export listings to data/listings.csv
python -m scraper export

# Train the model
python -m ml.train

# Run the API (dev)
uvicorn api:app --reload

# Run tests
pytest
```

Set `GOOGLE_MAPS_API_KEY` and `MAPS_MAP_ID` environment variables to enable the map panel.
