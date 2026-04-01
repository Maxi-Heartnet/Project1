# Design: Santo Domingo House Price Chatbot

**Date:** 2026-04-01
**Type:** Personal learning project
**Stack:** Python (scikit-learn, pandas, BeautifulSoup, joblib)

## Overview

A CLI chatbot that predicts house sale prices in Santo Domingo, Dominican Republic. The user answers a sequence of questions about a property; the chatbot returns an estimated price range in USD. Price estimation is powered by a supervised learning regression model trained on real listings scraped from local real estate sites.

## Architecture

The system has three sequential phases:

```
[1. Data Pipeline]  →  [2. ML Model]  →  [3. Chatbot CLI]
   Scrape listings      Train regressor     Collect features
   Clean & store CSV    Save model file     Query model
                        Evaluate accuracy   Display prediction
```

### Components

| Component | File(s) | Responsibility |
|---|---|---|
| Scraper | `scraper/scraper.py` | Scrapes listings from Corotos.com.do, outputs `data/listings.csv` |
| Data prep | `ml/prepare.py` | Cleans data, encodes categoricals, engineers features, splits train/test |
| Model | `ml/train.py`, `ml/model.pkl` | Trains Random Forest regressor, saves model + encoder, prints evaluation |
| Chatbot | `chatbot/chat.py` | CLI loop: collects features, loads model, predicts and displays price range |

## Data Pipeline

**Scraping target:** Corotos.com.do — "Inmuebles > Casas y Apartamentos", filtered to Santo Domingo.

**Features scraped per listing:**
- Price (USD)
- Sector/neighborhood (Piantini, Naco, Bella Vista, Evaristo Morales, Los Cacicazgos, etc.)
- Property type (apartment, house, villa)
- Bedrooms, bathrooms
- Floor area (m²)
- Parking spots
- Floor level (for apartments)

**Scraping approach:** `requests` + `BeautifulSoup`, paginating through listing pages. Raw HTML is cached locally so re-parsing doesn't require re-scraping. Final output: `data/listings.csv`.

**Data preparation (`ml/prepare.py`):**
- Drop rows missing price or area
- Remove price outliers (below $10,000 or above $5,000,000 USD — likely data errors)
- One-hot encode sector and property type
- Save the fitted encoder alongside the model so the chatbot uses identical encoding
- 80/20 train/test split

**Expected dataset size:** 500–2,000 listings.

## ML Model

**Algorithm:** `RandomForestRegressor` from scikit-learn.

Chosen because it handles mixed numeric/categorical features well, requires minimal tuning for a first working model, and produces feature importance scores that reveal which factors most drive Santo Domingo prices.

**Training (`ml/train.py`):**
1. Load `data/listings.csv`, run through `prepare.py`
2. Train `RandomForestRegressor` on training split
3. Evaluate on test split — report MAE and R²
4. Save a dict `{"model": ..., "encoder": ..., "feature_names": ...}` to `ml/model.pkl` via `joblib`

**Supervised learning framing:** each listing is a labeled training example — features are the house attributes, the label is the sale price. The model learns the mapping `(features) → price` from this data.

**Price range output:** derived from the spread of predictions across individual trees in the forest, giving a natural uncertainty estimate without extra code.

**Stretch goal:** compare against a Linear Regression baseline and a Gradient Boosting model to observe trade-offs in practice.

## Chatbot CLI

**Entry point:** `python chatbot/chat.py`

**Conversation flow:**
```
Welcome! I'll estimate a house price in Santo Domingo.

Property type: apartment or house? > apartment
Sector (e.g. Piantini, Naco, Bella Vista)? > Piantini
Bedrooms? > 3
Bathrooms? > 2
Area in m²? > 120
Parking spots? > 1
Floor level (apartments only)? > 5

Estimated price: $180,000 – $220,000 USD
```

**Implementation details:**
- Loads `ml/model.pkl` once at startup
- Applies the same encoder saved during training — guarantees consistent feature encoding
- Floor level question is skipped when property type is "house"
- Input validation re-prompts on invalid entries (non-numeric values, blank inputs)
- Unrecognized sectors fall back to a "Santo Domingo general" encoding with a printed warning
- The chatbot can be re-run indefinitely; type `exit` to quit

## File Structure

```
Project1/
├── scraper/
│   └── scraper.py          # Web scraper
├── ml/
│   ├── prepare.py          # Data cleaning + feature engineering
│   ├── train.py            # Model training + evaluation
│   └── model.pkl           # Saved model + encoder (generated, not committed)
├── data/
│   ├── raw/                # Cached HTML pages
│   └── listings.csv        # Cleaned dataset (generated, not committed)
├── chatbot/
│   └── chat.py             # CLI chatbot
└── requirements.txt        # pandas, scikit-learn, requests, beautifulsoup4, joblib
```

## Running the System

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Collect data
python scraper/scraper.py

# 3. Train the model
python ml/train.py

# 4. Run the chatbot
python chatbot/chat.py
```
