import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from chatbot.chat import predict_range, predict_tier
from ml.prepare import ALL_FEATURES

MODEL_PATH = Path(__file__).parent / 'ml' / 'model.pkl'

logger = logging.getLogger(__name__)
model_store = {}

_PAGE_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Santo Domingo House Price Estimator</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: system-ui, -apple-system, sans-serif;
  background: #f0f2f5;
  color: #222;
  min-height: 100vh;
  padding: 1.5rem 1rem;
}
main { max-width: 640px; margin: 0 auto; }
h1 { font-size: 1.35rem; font-weight: 700; margin-bottom: 1.25rem; color: #111; }
.card {
  background: #fff;
  border-radius: 10px;
  padding: 1.5rem;
  box-shadow: 0 1px 5px rgba(0,0,0,.09);
}
.form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
@media (max-width: 479px) { .form-grid { grid-template-columns: 1fr; } }
.field { display: flex; flex-direction: column; gap: 4px; }
label { font-size: .85rem; font-weight: 600; color: #444; }
input, select {
  width: 100%;
  padding: .5rem .7rem;
  border: 1px solid #ccc;
  border-radius: 6px;
  font-size: 1rem;
  color: #222;
  background: #fff;
}
input:focus, select:focus {
  outline: 2px solid #1a6fbd;
  outline-offset: 1px;
  border-color: #1a6fbd;
}
.field-error { font-size: .78rem; color: #c0392b; min-height: 1.1em; }
.submit-row { margin-top: 1.25rem; }
#submit-btn {
  width: 100%;
  padding: .65rem 1rem;
  background: #1a6fbd;
  color: #fff;
  border: none;
  border-radius: 6px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
}
#submit-btn:hover:not(:disabled) { background: #155a9e; }
#submit-btn:disabled { opacity: .6; cursor: not-allowed; }
#results {
  margin-top: 1.25rem;
  background: #fff;
  border-radius: 10px;
  padding: 1.5rem;
  box-shadow: 0 1px 5px rgba(0,0,0,.09);
  color: #666;
  min-height: 4rem;
}
.price-headline { font-size: 1.6rem; font-weight: 700; color: #111; margin-bottom: .7rem; }
.badge {
  display: inline-block;
  padding: .25rem .8rem;
  border-radius: 99px;
  color: #fff;
  font-size: .85rem;
  font-weight: 700;
  margin-bottom: 1rem;
}
.badge-Budget    { background: #2e7d32; }
.badge-MidRange  { background: #1a6fbd; }
.badge-Luxury    { background: #c9900a; }
.tier-stats {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: .3rem .9rem;
  font-size: .9rem;
}
.stat-label { color: #666; }
.stat-value { color: #222; font-weight: 500; }
.result-error { color: #c0392b; font-weight: 500; }
</style>
</head>
<body>
<main>
  <h1>Santo Domingo House Price Estimator</h1>
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
    </form>
  </div>
  <div id="results" aria-live="polite">Fill in the form above to see a price estimate.</div>
</main>
<script>
(function () {
  'use strict';
  var SECTORS = __SECTORS__;

  // Populate sector datalist
  var dl = document.getElementById('sector-list');
  SECTORS.forEach(function (s) {
    var opt = document.createElement('option');
    opt.value = s;
    dl.appendChild(opt);
  });

  // Validation rules
  function validateSector(v) {
    return v.trim() ? null : 'Please enter a sector or neighborhood.';
  }
  function validatePropertyType(v) {
    return v ? null : 'Please select a property type.';
  }
  function validateBedrooms(v) {
    var n = parseInt(v, 10);
    return (v !== '' && !isNaN(n) && n >= 1) ? null
      : 'Enter a whole number of bedrooms (1 or more).';
  }
  function validateArea(v) {
    var n = parseFloat(v);
    return (v !== '' && !isNaN(n) && n > 0) ? null
      : 'Enter a positive area in m\u00b2.';
  }

  var fields = [
    { id: 'sector',        validate: validateSector },
    { id: 'property_type', validate: validatePropertyType },
    { id: 'bedrooms',      validate: validateBedrooms },
    { id: 'area_m2',       validate: validateArea },
  ];

  function setError(fieldId, msg) {
    var el = document.getElementById(fieldId + '-error');
    if (el) el.textContent = msg || '';
  }

  // Blur validation
  fields.forEach(function (f) {
    var el = document.getElementById(f.id);
    if (!el) return;
    el.addEventListener('blur', function () {
      setError(f.id, f.validate(el.value) || '');
    });
  });

  // Formatting helpers
  function fmtFull(n) {
    return n.toLocaleString('en-US', { maximumFractionDigits: 0 });
  }
  function fmtK(n) {
    return '$' + Math.round(n / 1000) + 'k';
  }

  // Whitelist for tier badge CSS classes (prevents raw API value in className)
  var TIER_CLASS = {
    'Budget':    'badge-Budget',
    'Mid-Range': 'badge-MidRange',
    'Luxury':    'badge-Luxury',
  };

  function clearResults() {
    var r = document.getElementById('results');
    while (r.firstChild) r.removeChild(r.firstChild);
    return r;
  }

  function renderResults(data) {
    var r = clearResults();

    var headline = document.createElement('div');
    headline.className = 'price-headline';
    headline.textContent = '$' + fmtFull(data.price_low) + ' \u2013 $' + fmtFull(data.price_high) + ' USD';
    r.appendChild(headline);

    var badge = document.createElement('span');
    badge.className = 'badge ' + (TIER_CLASS[data.market_tier] || 'badge-MidRange');
    badge.textContent = data.market_tier;
    r.appendChild(badge);

    var s = data.tier_stats;
    var statsEl = document.createElement('div');
    statsEl.className = 'tier-stats';

    var rows = [
      ['Typical price range', fmtK(s.price_p10) + ' \u2013 ' + fmtK(s.price_p90) + ' USD'],
      ['Typical area',        Math.round(s.area_p10) + '\u2013' + Math.round(s.area_p90) + ' m\u00b2'],
      ['Typical bedrooms',    Math.round(s.beds_p10) + '\u2013' + Math.round(s.beds_p90) + ' beds'],
    ];
    rows.forEach(function (row) {
      var lbl = document.createElement('span');
      lbl.className = 'stat-label';
      lbl.textContent = row[0];
      var val = document.createElement('span');
      val.className = 'stat-value';
      val.textContent = row[1];
      statsEl.appendChild(lbl);
      statsEl.appendChild(val);
    });
    r.appendChild(statsEl);
  }

  function renderError(msg) {
    var r = clearResults();
    var p = document.createElement('p');
    p.className = 'result-error';
    p.textContent = msg;
    r.appendChild(p);
  }

  var form = document.getElementById('predict-form');
  var btn  = document.getElementById('submit-btn');

  form.addEventListener('submit', function (e) {
    e.preventDefault();

    // Validate all
    var valid = true;
    fields.forEach(function (f) {
      var el  = document.getElementById(f.id);
      var err = f.validate(el.value);
      setError(f.id, err || '');
      if (err) valid = false;
    });
    if (!valid) return;

    // Loading state
    btn.disabled = true;
    btn.textContent = 'Estimating...';
    var r = clearResults();
    var loadingMsg = document.createElement('p');
    loadingMsg.textContent = 'Getting your estimate\u2026';
    r.appendChild(loadingMsg);

    var payload = {
      sector:        document.getElementById('sector').value.trim(),
      property_type: document.getElementById('property_type').value,
      bedrooms:      parseInt(document.getElementById('bedrooms').value, 10),
      area_m2:       parseFloat(document.getElementById('area_m2').value),
    };

    fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    .then(function (response) {
      if (!response.ok) { throw { isServer: true }; }
      return response.json();
    })
    .then(function (data) {
      renderResults(data);
    })
    .catch(function (err) {
      if (err && err.isServer) {
        renderError('Something went wrong. Please try again.');
      } else {
        renderError('Could not reach the server. Check your connection and try again.');
      }
      btn.disabled = false;
      btn.textContent = 'Estimate Price';
    });
  });
}());
</script>
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
    yield
    model_store.clear()


app = FastAPI(
    title="Santo Domingo House Price API",
    description="Predicts house prices and market tier for properties in Santo Domingo, DR.",
    lifespan=lifespan,
)


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
    html = _PAGE_HTML.replace('__SECTORS__', json.dumps(sectors).replace('</', '\\/'))
    return HTMLResponse(content=html)


@app.post('/predict', response_model=PredictResponse)
def predict(body: PredictRequest):
    sector = body.sector.title()
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
