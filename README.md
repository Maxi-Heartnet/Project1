# Santo Domingo Real Estate Price Predictor

A web scraper + ML pipeline that collects real estate listings from DR sites and trains a price prediction model, served via a FastAPI endpoint.

## Developer Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# One-time: install Playwright browser binaries (required for Phase 2 scrapers)
playwright install chromium
```

> **Note:** `playwright install chromium` only needs to be run locally. EC2 runs the API only — browser binaries are not required there.

## Usage

```bash
# Scrape all sites into data/listings.db
python -m scraper

# Export listings to data/listings.csv
python -m scraper export

# Train the model
python -m ml.train

# Run the API (dev)
uvicorn api:app --reload
```

## Running Tests

```bash
pytest
```

## Project Structure

```
scraper/        Web scrapers (Phase 1: requests+BS4, Phase 2: Playwright)
ml/             Training pipeline
api.py          FastAPI prediction endpoint
data/           listings.db, listings.csv (gitignored)
docs/plans/     Implementation plans
docs/solutions/ Documented solutions to past bugs
terraform/      EC2 deployment (AWS)
```
