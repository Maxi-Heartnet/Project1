# Santo Domingo House Price Chatbot — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI chatbot that collects house features and predicts a USD price range for Santo Domingo real estate using a Random Forest regression model trained on scraped listings.

**Architecture:** A scraper collects listings from Corotos.com.do into a CSV; a data prep module cleans and encodes the data; a training script fits a RandomForestRegressor and saves a `model.pkl` artifact; a CLI chatbot loads the artifact and runs predictions. All phases are independent — prep/train/chatbot can be developed and tested with synthetic data before scraping.

**Tech Stack:** Python 3.10+, pandas, scikit-learn >= 1.2, requests, beautifulsoup4, joblib, pytest

---

## File Map

| File | Role |
|---|---|
| `requirements.txt` | All dependencies |
| `.gitignore` | Ignore generated data and model files |
| `ml/__init__.py` | Package marker |
| `ml/prepare.py` | Clean CSV, encode features, split train/test — returns encoded arrays + fitted preprocessor |
| `ml/train.py` | Load prepared data, train RandomForest, print metrics, save `ml/model.pkl` |
| `scraper/__init__.py` | Package marker |
| `scraper/scraper.py` | Fetch + cache Corotos pages, parse listing cards, write `data/listings.csv` |
| `chatbot/__init__.py` | Package marker |
| `chatbot/chat.py` | CLI loop — ask questions, build feature DataFrame, call model, print price range |
| `tests/test_prepare.py` | Unit tests for clean() and load_and_prepare() with synthetic DataFrames |
| `tests/test_train.py` | Integration test: train on small synthetic CSV, verify artifact structure |
| `tests/test_scraper.py` | Unit tests for parse_listing() and parse_price_usd() with fixture HTML |
| `tests/test_chat.py` | Unit tests for ask_string(), ask_number(), predict_range(), collect_features() |

---

## Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `ml/__init__.py`, `scraper/__init__.py`, `chatbot/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
pandas
scikit-learn>=1.2
requests
beautifulsoup4
joblib
pytest
```

- [ ] **Step 2: Create .gitignore**

```
data/listings.csv
data/raw/
ml/model.pkl
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 3: Create package markers**

Create empty `ml/__init__.py`, `scraper/__init__.py`, `chatbot/__init__.py`, `tests/__init__.py`.

- [ ] **Step 4: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: All packages install without error.

- [ ] **Step 5: Commit**

```bash
git init
git add requirements.txt .gitignore ml/__init__.py scraper/__init__.py chatbot/__init__.py tests/__init__.py
git commit -m "chore: project scaffolding"
```

---

## Task 2: Data Preparation Module

**Files:**
- Create: `ml/prepare.py`
- Create: `tests/test_prepare.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_prepare.py`:

```python
import pandas as pd
import pytest
from ml.prepare import clean, load_and_prepare


def make_df(rows):
    defaults = {
        'bedrooms': 3, 'bathrooms': 2, 'area_m2': 100, 'parking': 1,
        'floor_level': None, 'sector': 'Piantini', 'property_type': 'apartment',
        'price': 150_000,
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def test_clean_drops_missing_price():
    df = make_df([{}, {'price': None}])
    result = clean(df)
    assert len(result) == 1


def test_clean_drops_missing_area():
    df = make_df([{}, {'area_m2': None}])
    result = clean(df)
    assert len(result) == 1


def test_clean_removes_price_outliers():
    df = make_df([{}, {'price': 5_000}, {'price': 6_000_000}])
    result = clean(df)
    assert len(result) == 1
    assert result.iloc[0]['price'] == 150_000


def test_clean_fills_floor_level_with_zero():
    df = make_df([{'floor_level': None}])
    result = clean(df)
    assert result.iloc[0]['floor_level'] == 0


def test_load_and_prepare_shapes(tmp_path):
    rows = [
        {'bedrooms': (i % 4) + 1, 'bathrooms': 2, 'area_m2': 80 + i * 10,
         'parking': 1, 'floor_level': i % 5, 'sector': 'Naco',
         'property_type': 'apartment', 'price': 100_000 + i * 10_000}
        for i in range(20)
    ]
    csv = tmp_path / "listings.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    X_train, X_test, y_train, y_test, preprocessor = load_and_prepare(str(csv))
    assert X_train.shape[0] == 16   # 80% of 20
    assert X_test.shape[0] == 4
    assert X_train.shape[1] == X_test.shape[1]
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/test_prepare.py -v`
Expected: `ImportError` or `ModuleNotFoundError` — `ml.prepare` does not exist yet.

- [ ] **Step 3: Implement ml/prepare.py**

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

NUMERIC_FEATURES = ['bedrooms', 'bathrooms', 'area_m2', 'parking', 'floor_level']
CATEGORICAL_FEATURES = ['sector', 'property_type']
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
PRICE_MIN = 10_000
PRICE_MAX = 5_000_000


def clean(df):
    df = df.dropna(subset=['price', 'area_m2'])
    df = df[(df['price'] >= PRICE_MIN) & (df['price'] <= PRICE_MAX)].copy()
    df['floor_level'] = df['floor_level'].fillna(0)
    for col in ['bedrooms', 'bathrooms', 'parking']:
        df[col] = df[col].fillna(df[col].median())
    return df


def build_preprocessor():
    return ColumnTransformer(transformers=[
        ('num', 'passthrough', NUMERIC_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES),
    ])


def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)
    df = clean(df)
    X = df[ALL_FEATURES]
    y = df['price'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    return (
        preprocessor.transform(X_train),
        preprocessor.transform(X_test),
        y_train,
        y_test,
        preprocessor,
    )
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest tests/test_prepare.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ml/prepare.py tests/test_prepare.py
git commit -m "feat: data preparation module with cleaning and encoding"
```

---

## Task 3: Model Training

**Files:**
- Create: `ml/train.py`
- Create: `tests/test_train.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_train.py`:

```python
import random
import joblib
import pandas as pd
import pytest
from ml.train import train


def make_csv(tmp_path, n=40):
    rows = [
        {
            'bedrooms': random.randint(1, 5),
            'bathrooms': random.randint(1, 3),
            'area_m2': random.randint(50, 300),
            'parking': random.randint(0, 3),
            'floor_level': random.randint(0, 10),
            'sector': random.choice(['Piantini', 'Naco', 'Bella Vista']),
            'property_type': random.choice(['apartment', 'house']),
            'price': random.randint(80_000, 500_000),
        }
        for _ in range(n)
    ]
    csv = tmp_path / "listings.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return str(csv)


def test_train_saves_artifact_with_required_keys(tmp_path):
    csv = make_csv(tmp_path)
    model_path = str(tmp_path / "model.pkl")
    train(csv, model_path=model_path)
    artifact = joblib.load(model_path)
    assert 'model' in artifact
    assert 'encoder' in artifact


def test_train_model_produces_positive_predictions(tmp_path):
    csv = make_csv(tmp_path)
    model_path = str(tmp_path / "model.pkl")
    train(csv, model_path=model_path)
    artifact = joblib.load(model_path)
    model = artifact['model']
    encoder = artifact['encoder']
    sample = pd.DataFrame([{
        'bedrooms': 3, 'bathrooms': 2, 'area_m2': 120,
        'parking': 1, 'floor_level': 5,
        'sector': 'Piantini', 'property_type': 'apartment',
    }])
    pred = model.predict(encoder.transform(sample))
    assert pred[0] > 0
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/test_train.py -v`
Expected: `ImportError` — `ml.train` does not exist yet.

- [ ] **Step 3: Implement ml/train.py**

```python
import sys
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from ml.prepare import load_and_prepare

MODEL_PATH = 'ml/model.pkl'


def train(csv_path, model_path=MODEL_PATH):
    X_train, X_test, y_train, y_test, preprocessor = load_and_prepare(csv_path)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"MAE: ${mean_absolute_error(y_test, y_pred):,.0f}")
    print(f"R²:  {r2_score(y_test, y_pred):.3f}")
    joblib.dump({"model": model, "encoder": preprocessor}, model_path)
    print(f"Model saved to {model_path}")


if __name__ == '__main__':
    csv = sys.argv[1] if len(sys.argv) > 1 else 'data/listings.csv'
    train(csv)
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest tests/test_train.py -v`
Expected: Both tests PASS. Training on 40 rows may give a low R² — that is expected with synthetic data; the tests only check artifact structure and positive predictions.

- [ ] **Step 5: Commit**

```bash
git add ml/train.py tests/test_train.py
git commit -m "feat: model training with RandomForest, saves model.pkl artifact"
```

---

## Task 4: Web Scraper

**Files:**
- Create: `scraper/scraper.py`
- Create: `tests/test_scraper.py`

> **Before coding:** Open `https://www.corotos.com.do/p/inmuebles/casas-y-apartamentos` in a browser,
> right-click a listing card and choose "Inspect". Identify the correct CSS selectors for:
> - The listing card container
> - Price element
> - Location/sector element
> - Title element
> - Detail list items (bedrooms, bathrooms, area, parking, floor)
>
> Update the `*_SELECTOR` constants at the top of `scraper.py` to match, and update
> `SAMPLE_CARD_HTML` in `test_scraper.py` to use those same class names.

- [ ] **Step 1: Write failing tests**

Create `tests/test_scraper.py` — the fixture HTML uses the default selectors in `scraper.py`; update it if you changed them after inspecting the site:

```python
from bs4 import BeautifulSoup
from scraper.scraper import parse_listing, parse_price_usd

# Minimal fixture matching the CARD_SELECTOR, PRICE_SELECTOR, etc. in scraper.py
SAMPLE_CARD_HTML = """
<div class="ad-card">
  <h2 class="title">Apartamento en Venta en Piantini</h2>
  <span class="price">US$185,000</span>
  <span class="location">Piantini</span>
  <ul class="ad-details">
    <li>3 Habitaciones</li>
    <li>2 Baños</li>
    <li>120 m²</li>
    <li>1 Parking</li>
    <li>Piso 5</li>
  </ul>
</div>
"""


def card():
    return BeautifulSoup(SAMPLE_CARD_HTML, 'html.parser').select_one('div.ad-card')


def test_parse_price_usd_valid():
    assert parse_price_usd("US$185,000") == 185000


def test_parse_price_usd_rejects_peso():
    assert parse_price_usd("RD$3,500,000") is None


def test_parse_listing_price():
    assert parse_listing(card())['price'] == 185000


def test_parse_listing_sector():
    assert parse_listing(card())['sector'] == 'Piantini'


def test_parse_listing_area():
    assert parse_listing(card())['area_m2'] == 120


def test_parse_listing_property_type():
    assert parse_listing(card())['property_type'] == 'apartment'


def test_parse_listing_bedrooms():
    assert parse_listing(card())['bedrooms'] == 3
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/test_scraper.py -v`
Expected: `ImportError` — `scraper.scraper` does not exist yet.

- [ ] **Step 3: Implement scraper/scraper.py**

```python
import csv
import os
import re
import time

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.corotos.com.do/p/inmuebles/casas-y-apartamentos"
RAW_DIR = "data/raw"
OUTPUT_CSV = "data/listings.csv"
FIELDNAMES = ['price', 'sector', 'property_type', 'bedrooms', 'bathrooms',
              'area_m2', 'parking', 'floor_level']

# CSS selectors — verify by inspecting the site and update if needed
CARD_SELECTOR = "div.ad-card"
PRICE_SELECTOR = "span.price"
TITLE_SELECTOR = "h2.title"
DETAILS_SELECTOR = "ul.ad-details li"
LOCATION_SELECTOR = "span.location"


def parse_price_usd(text):
    """Return integer USD price, or None if not a USD listing."""
    text = text.strip()
    upper = text.upper()
    if 'RD$' in upper or 'DOP' in upper:
        return None
    if 'US$' not in upper and 'USD' not in upper and '$' not in text:
        return None
    numbers = re.findall(r'\d+', text)
    return int(''.join(numbers)) if numbers else None


def _first_int(text):
    numbers = re.findall(r'\d+', text)
    return int(numbers[0]) if numbers else None


def parse_listing(card):
    """Parse a BeautifulSoup Tag (one listing card) into a dict."""
    price_el = card.select_one(PRICE_SELECTOR)
    price = parse_price_usd(price_el.get_text()) if price_el else None

    location_el = card.select_one(LOCATION_SELECTOR)
    sector = location_el.get_text(strip=True) if location_el else None

    title_el = card.select_one(TITLE_SELECTOR)
    title = title_el.get_text(strip=True).lower() if title_el else ''
    if 'apartamento' in title or 'apartment' in title:
        property_type = 'apartment'
    elif 'casa' in title or 'house' in title:
        property_type = 'house'
    else:
        property_type = 'apartment'

    details = [el.get_text(strip=True) for el in card.select(DETAILS_SELECTOR)]
    bedrooms = bathrooms = area_m2 = parking = floor_level = None
    for d in details:
        dl = d.lower()
        if bedrooms is None and ('hab' in dl or 'bedroom' in dl):
            bedrooms = _first_int(d)
        elif bathrooms is None and ('baño' in dl or 'bathroom' in dl):
            bathrooms = _first_int(d)
        elif area_m2 is None and 'm²' in dl:
            area_m2 = _first_int(d)
        elif parking is None and ('parking' in dl or 'garage' in dl or 'garaje' in dl):
            parking = _first_int(d)
        elif floor_level is None and ('piso' in dl or 'floor' in dl):
            floor_level = _first_int(d)

    return {
        'price': price, 'sector': sector, 'property_type': property_type,
        'bedrooms': bedrooms, 'bathrooms': bathrooms, 'area_m2': area_m2,
        'parking': parking, 'floor_level': floor_level,
    }


def fetch_page(page_num, use_cache=True):
    cache_path = os.path.join(RAW_DIR, f"page_{page_num}.html")
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return f.read()
    resp = requests.get(
        BASE_URL,
        params={'page': page_num},
        headers={'User-Agent': 'Mozilla/5.0'},
        timeout=15,
    )
    resp.raise_for_status()
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(cache_path, 'w', encoding='utf-8') as f:
        f.write(resp.text)
    return resp.text


def scrape(max_pages=30, use_cache=True):
    listings = []
    for page_num in range(1, max_pages + 1):
        html = fetch_page(page_num, use_cache=use_cache)
        soup = BeautifulSoup(html, 'html.parser')
        cards = soup.select(CARD_SELECTOR)
        if not cards:
            print(f"Page {page_num}: no cards found — stopping.")
            break
        for card in cards:
            row = parse_listing(card)
            if row['price'] and row['area_m2']:
                listings.append(row)
        print(f"Page {page_num}: {len(cards)} cards | total so far: {len(listings)}")
        time.sleep(1)
    return listings


def save_csv(listings, path=OUTPUT_CSV):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(listings)
    print(f"Saved {len(listings)} listings to {path}")


if __name__ == '__main__':
    save_csv(scrape())
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest tests/test_scraper.py -v`
Expected: All 7 tests PASS.

If any parsing test fails, inspect the fixture HTML and `parse_listing` to find the mismatch — do not change the selector constants until you have verified them against the live site.

- [ ] **Step 5: Run the real scraper**

Run: `python scraper/scraper.py`

Watch the output — if you see "no cards found" on page 1, the `CARD_SELECTOR` is wrong. Open `data/raw/page_1.html` in a browser or text editor, find the listing card elements, note their class names, update `CARD_SELECTOR` (and the other selectors) in `scraper.py`, then update `SAMPLE_CARD_HTML` in `tests/test_scraper.py` to match, and re-run tests.

Expected when working: output like `Page 1: 20 cards | total so far: 18` repeating for multiple pages, then `Saved NNN listings to data/listings.csv`.

- [ ] **Step 6: Commit**

```bash
git add scraper/scraper.py tests/test_scraper.py
git commit -m "feat: Corotos scraper with HTML caching and CSV output"
```

---

## Task 5: CLI Chatbot

**Files:**
- Create: `chatbot/chat.py`
- Create: `tests/test_chat.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_chat.py`:

```python
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from chatbot.chat import ask_string, ask_number, collect_features, predict_range


def test_ask_string_accepts_valid_choice():
    with patch('builtins.input', return_value='apartment'):
        assert ask_string("p > ", choices=['apartment', 'house']) == 'apartment'


def test_ask_string_rejects_then_accepts():
    with patch('builtins.input', side_effect=['villa', 'house']):
        assert ask_string("p > ", choices=['apartment', 'house']) == 'house'


def test_ask_string_no_choices_returns_value():
    with patch('builtins.input', return_value='Piantini'):
        assert ask_string("p > ") == 'Piantini'


def test_ask_number_valid_int():
    with patch('builtins.input', return_value='3'):
        assert ask_number("p > ", dtype=int, min_val=1) == 3


def test_ask_number_rejects_non_numeric_then_accepts():
    with patch('builtins.input', side_effect=['abc', '120']):
        assert ask_number("p > ", dtype=float, min_val=1) == 120.0


def test_ask_number_rejects_below_min():
    with patch('builtins.input', side_effect=['0', '1']):
        assert ask_number("p > ", dtype=int, min_val=1) == 1


def test_predict_range_low_lte_high():
    mock_tree = MagicMock()
    mock_tree.predict.return_value = np.array([150_000.0])
    mock_model = MagicMock()
    mock_model.estimators_ = [mock_tree] * 20

    mock_encoder = MagicMock()
    mock_encoder.transform.return_value = np.zeros((1, 8))

    features_df = pd.DataFrame([{
        'bedrooms': 3, 'bathrooms': 2, 'area_m2': 120,
        'parking': 1, 'floor_level': 5,
        'sector': 'Piantini', 'property_type': 'apartment',
    }])
    low, high = predict_range(mock_model, mock_encoder, features_df)
    assert low <= high
    assert low > 0


def test_collect_features_skips_floor_level_for_house():
    with patch('builtins.input', side_effect=['house', 'Naco', '3', '2', '150', '1']):
        features = collect_features()
    assert features['property_type'] == 'house'
    assert features['floor_level'] == 0


def test_collect_features_asks_floor_level_for_apartment():
    with patch('builtins.input', side_effect=['apartment', 'Piantini', '3', '2', '120', '1', '5']):
        features = collect_features()
    assert features['floor_level'] == 5
    assert features['sector'] == 'Piantini'
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/test_chat.py -v`
Expected: `ImportError` — `chatbot.chat` does not exist yet.

- [ ] **Step 3: Implement chatbot/chat.py**

```python
import sys
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = 'ml/model.pkl'


def ask_string(prompt, choices=None):
    while True:
        raw = input(prompt).strip()
        if raw.lower() == 'exit':
            print("Goodbye!")
            sys.exit(0)
        if not raw:
            print("  Cannot be blank. Try again.")
            continue
        if choices and raw.lower() not in [c.lower() for c in choices]:
            print(f"  Enter one of: {', '.join(choices)}. Try again.")
            continue
        return raw.lower() if choices else raw


def ask_number(prompt, dtype=int, min_val=0):
    while True:
        raw = input(prompt).strip()
        if raw.lower() == 'exit':
            print("Goodbye!")
            sys.exit(0)
        try:
            val = dtype(raw)
        except ValueError:
            print("  Enter a number. Try again.")
            continue
        if val < min_val:
            print(f"  Must be >= {min_val}. Try again.")
            continue
        return val


def predict_range(model, encoder, features_df):
    X = encoder.transform(features_df)
    tree_preds = np.array([tree.predict(X)[0] for tree in model.estimators_])
    return np.percentile(tree_preds, 10), np.percentile(tree_preds, 90)


def collect_features():
    prop_type = ask_string("Property type (apartment/house)? > ", choices=['apartment', 'house'])
    sector = ask_string("Sector (e.g. Piantini, Naco, Bella Vista)? > ")
    bedrooms = ask_number("Bedrooms? > ", dtype=int, min_val=1)
    bathrooms = ask_number("Bathrooms? > ", dtype=float, min_val=0.5)
    area_m2 = ask_number("Area in m²? > ", dtype=float, min_val=1)
    parking = ask_number("Parking spots? > ", dtype=int, min_val=0)
    floor_level = 0
    if prop_type == 'apartment':
        floor_level = ask_number("Floor level? > ", dtype=int, min_val=0)
    return {
        'bedrooms': bedrooms, 'bathrooms': bathrooms, 'area_m2': area_m2,
        'parking': parking, 'floor_level': floor_level,
        'sector': sector.title(), 'property_type': prop_type,
    }


def main(model_path=MODEL_PATH):
    artifact = joblib.load(model_path)
    model = artifact['model']
    encoder = artifact['encoder']

    print("\nWelcome! I'll estimate a house price in Santo Domingo.")
    print("Type 'exit' at any prompt to quit.\n")

    while True:
        features = collect_features()
        low, high = predict_range(model, encoder, pd.DataFrame([features]))
        print(f"\nEstimated price: ${low:,.0f} – ${high:,.0f} USD\n")
        again = ask_string("Estimate another? (yes/no) > ", choices=['yes', 'no'])
        if again == 'no':
            print("Goodbye!")
            break


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest tests/test_chat.py -v`
Expected: All 9 tests PASS.

- [ ] **Step 5: Run full end-to-end check**

Prerequisites: `data/listings.csv` exists (from Task 4 scraper run), model not yet trained.

```bash
python ml/train.py
```

Expected output (numbers will vary):
```
MAE: $XX,XXX
R²:  0.XXX
Model saved to ml/model.pkl
```

Then run the chatbot:
```bash
python chatbot/chat.py
```

Walk through the prompts with a real example: apartment, Piantini, 3 bed, 2 bath, 120 m², 1 parking, floor 5. Verify a USD price range is printed.

- [ ] **Step 6: Run full test suite**

Run: `pytest -v`
Expected: All tests PASS.

- [ ] **Step 7: Commit**

```bash
git add chatbot/chat.py tests/test_chat.py
git commit -m "feat: CLI chatbot with price range prediction"
```

---

## Running the Complete System

```bash
pip install -r requirements.txt   # once
python scraper/scraper.py          # collect data → data/listings.csv
python ml/train.py                 # train model → ml/model.pkl
python chatbot/chat.py             # run chatbot
pytest -v                          # run all tests
```

All commands must be run from the project root directory (`Project1/`).
