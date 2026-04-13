import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from chatbot.chat import predict_range, predict_tier
from ml.prepare import ALL_FEATURES

MODEL_PATH = Path(__file__).parent / 'ml' / 'model.pkl'
SECTOR_COORDS_PATH = Path(__file__).parent / 'static' / 'sector_coords.json'

MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY', '')
MAPS_MAP_ID = os.getenv('MAPS_MAP_ID', '')

logger = logging.getLogger(__name__)
model_store = {}

_PAGE_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Santo Domingo House Price Estimator</title>
<link rel="stylesheet" href="/static/style.css">
</head>
<body>
<main>
  <header class="page-header">
    <h1>Santo Domingo House Price Estimator</h1>
    <a href="/about" class="btn-secondary">About Me</a>
  </header>
  <div class="card">
    <form id="predict-form" novalidate>
      <div class="form-grid">
        <div class="field">
          <label for="sector">Sector</label>
          <input id="sector" name="sector" type="text" autocomplete="off"
                 list="sector-list" placeholder="e.g. Piantini, Naco">
          <datalist id="sector-list"></datalist>
          <span class="field-error" id="sector-error" role="alert"></span>
        </div>
        <div class="field">
          <label for="property_type">Property Type</label>
          <select id="property_type" name="property_type">
            <option value="">Select type</option>
            <option value="apartment">Apartment</option>
            <option value="house">House</option>
          </select>
          <span class="field-error" id="property_type-error" role="alert"></span>
        </div>
        <div class="field">
          <label for="bedrooms">Bedrooms</label>
          <input id="bedrooms" name="bedrooms" type="number" min="1" step="1" placeholder="e.g. 3">
          <span class="field-error" id="bedrooms-error" role="alert"></span>
        </div>
        <div class="field">
          <label for="area_m2">Area m\u00b2</label>
          <input id="area_m2" name="area_m2" type="number" min="0.1" step="0.1" placeholder="e.g. 120">
          <span class="field-error" id="area_m2-error" role="alert"></span>
        </div>
      </div>
      <div class="submit-row">
        <button type="submit" id="submit-btn">Estimate Price</button>
      </div>
      <div class="action-row">
        <button type="button" id="clear-btn" class="btn-secondary">Clear</button>
        <button type="button" id="fill-btn" class="btn-secondary">Fill randomly</button>
      </div>
    </form>
  </div>
  <div id="results" aria-live="polite">Fill in the form above to see a price estimate.</div>
  <div class="card" id="map-card">
    <div id="map-container">
      <p class="map-loading">Loading map\u2026</p>
    </div>
    <div id="map-fallback" hidden>Map unavailable \u2014 use the text field to enter a sector.</div>
  </div>
</main>
<script>var SECTORS = __SECTORS__; var SECTOR_COORDS = __SECTOR_COORDS__; var MAPS_MAP_ID = "__MAPS_MAP_ID__";</script>
<script src="/static/map.js"></script>
<script async defer src="https://maps.googleapis.com/maps/api/js?key=__MAPS_API_KEY__&callback=initMap"></script>
<script src="/static/main.js"></script>
</body>
</html>
"""

_ABOUT_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>About Me \u2014 Misael</title>
<link rel="stylesheet" href="/static/style.css">
</head>
<body>
<main>
  <header class="page-header">
    <h1>About Me</h1>
    <a href="/" class="btn-secondary">\u2190 Back to App</a>
  </header>
  <div class="card">
    <h2>Misael</h2>
    <p class="subtitle">Full Stack &amp; ML Developer</p>
    <p>Builder of end-to-end solutions spanning data science, backend APIs, web frontends, and
    cloud infrastructure; comfortable working across the full stack and deploying to production.</p>

    <h3>Skills</h3>
    <p><strong>Languages &amp; Markup</strong></p>
    <ul>
      <li>Python, JavaScript, HTML/CSS, HCL (Terraform)</li>
    </ul>
    <p><strong>ML &amp; Data</strong></p>
    <ul>
      <li>scikit-learn, pandas, numpy, joblib &mdash; model training, pipelines,
      ColumnTransformer, serialization</li>
    </ul>
    <p><strong>Backend</strong></p>
    <ul>
      <li>FastAPI, uvicorn, REST API design</li>
    </ul>
    <p><strong>Frontend</strong></p>
    <ul>
      <li>Vanilla JS, async fetch, form validation, responsive layouts</li>
    </ul>
    <p><strong>Cloud &amp; Infra</strong></p>
    <ul>
      <li>AWS EC2, Terraform (IaC), Nginx, Supervisor</li>
    </ul>
    <p><strong>Tooling</strong></p>
    <ul>
      <li>Git/GitHub, conventional commits, PR workflow, Claude Code (AI-assisted dev)</li>
    </ul>

    <h3>Featured Project</h3>
    <p>Santo Domingo House Price Predictor &mdash; end-to-end ML application: trained a Random
    Forest on Dominican Republic real estate data, exposed it via a FastAPI REST endpoint, built
    a self-contained prediction form, and deployed the whole stack to AWS EC2 with a single
    <code>terraform apply</code>.</p>
  </div>
</main>
</body>
</html>
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifact = joblib.load(MODEL_PATH)
    model_store['model'] = artifact['model']
    model_store['encoder'] = artifact['encoder']
    model_store['clusterer'] = artifact['clusterer']
    model_store['cluster_label_map'] = artifact['cluster_label_map']
    model_store['cluster_stats'] = artifact['cluster_stats']
    try:
        model_store['known_sectors'] = (
            artifact['encoder'].named_transformers_['cat'].categories_[0].tolist()
        )
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        logger.warning('Could not extract sector list from encoder: %s', exc)
        model_store['known_sectors'] = []
    try:
        with open(SECTOR_COORDS_PATH, encoding='utf-8') as f:
            model_store['sector_coords'] = json.load(f)
    except FileNotFoundError:
        logger.warning('sector_coords.json not found — map will show no pins')
        model_store['sector_coords'] = {}
    yield
    model_store.clear()


app = FastAPI(
    title="Santo Domingo House Price API",
    description="Predicts house prices and market tier for properties in Santo Domingo, DR.",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")


class PredictRequest(BaseModel):
    sector: str
    property_type: str
    bedrooms: int = Field(ge=1)
    area_m2: float = Field(gt=0)


class TierStats(BaseModel):
    price_p10: float
    price_p90: float
    area_p10: float
    area_p90: float
    beds_p10: float
    beds_p90: float


class PredictResponse(BaseModel):
    price_low: float
    price_high: float
    market_tier: str
    tier_stats: TierStats


@app.get('/', response_class=HTMLResponse)
def index() -> HTMLResponse:
    sectors = model_store.get('known_sectors') or []
    sector_coords = model_store.get('sector_coords') or {}
    html = (
        _PAGE_HTML
        .replace('__SECTORS__', json.dumps(sectors).replace('</', '\\/'))
        .replace('__SECTOR_COORDS__', json.dumps(sector_coords).replace('</', '\\/'))
        .replace('__MAPS_API_KEY__', MAPS_API_KEY)
        .replace('__MAPS_MAP_ID__', MAPS_MAP_ID)
    )
    return HTMLResponse(content=html)


@app.get('/about', response_class=HTMLResponse)
def about() -> HTMLResponse:
    return HTMLResponse(content=_ABOUT_HTML)


@app.post('/predict', response_model=PredictResponse)
def predict(body: PredictRequest):
    sector = body.sector.strip()
    property_type = body.property_type.lower()

    features = {
        'bedrooms': body.bedrooms,
        'area_m2': body.area_m2,
        'sector': sector,
        'property_type': property_type,
    }
    features_df = pd.DataFrame([features], columns=ALL_FEATURES)

    low, high = predict_range(model_store['model'], model_store['encoder'], features_df)

    tier_data = predict_tier(
        model_store['clusterer'],
        model_store['cluster_label_map'],
        model_store['cluster_stats'],
        bedrooms=body.bedrooms,
        area_m2=body.area_m2,
        low=float(low),
        high=float(high),
    )

    return PredictResponse(
        price_low=float(low),
        price_high=float(high),
        market_tier=tier_data['market_tier'],
        tier_stats=TierStats(**tier_data['tier_stats']),
    )


@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': 'model' in model_store}
