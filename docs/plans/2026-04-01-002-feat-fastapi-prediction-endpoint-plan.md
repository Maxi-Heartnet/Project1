---
title: "feat: Add FastAPI HTTP prediction endpoint"
type: feat
status: completed
date: 2026-04-01
origin: docs/brainstorms/2026-04-01-fastapi-endpoint-requirements.md
---

# feat: Add FastAPI HTTP prediction endpoint

## Overview

Expose the house price prediction and market segmentation model as a public HTTP API using FastAPI, deployable on an AWS EC2 Ubuntu instance. Any caller can POST property features and receive a price range estimate and market tier in JSON.

## Problem Frame

The existing chatbot is interactive and only usable on the local machine. This change wraps the same ML logic in an HTTP endpoint so that anyone with the server's IP address can query predictions via `curl`, a browser, or any HTTP client — without installing Python.
(see origin: docs/brainstorms/2026-04-01-fastapi-endpoint-requirements.md)

## Requirements Trace

- R1. `POST /predict` accepts property features and returns price range + market tier
- R2. Request body: `sector` (str), `property_type` (str), `bedrooms` (int ≥ 1), `area_m2` (float > 0)
- R3. Response body: `price_low`, `price_high`, `market_tier`, `tier_stats` (P10/P90 for price/area/beds)
- R4. Invalid or missing fields → HTTP 422 with descriptive error (automatic via Pydantic)
- R5. `GET /health` → HTTP 200 confirming server is alive and model is loaded
- R6. Interactive docs at `/docs` (Swagger UI, automatic in FastAPI)
- R7. Runs on EC2 Ubuntu port 8000, publicly accessible
- R8. Process survives SSH disconnect via Supervisor
- R9. Model loaded once at startup, not per request
- R10. Endpoint is public — no authentication
- R11. `sector` normalized to `.title()`, `property_type` to `.lower()` before encoding

## Scope Boundaries

- No HTTPS / custom domain — HTTP on port 8000 only
- No authentication
- No persistent request logging or database
- No auto-retraining from the API
- No changes to scraper or data pipeline
- No rate limiting
- `chatbot/chat.py` CLI remains unchanged

## Context & Research

### Relevant Code and Patterns

- `chatbot/chat.py` — `predict_range(model, encoder, features_df)` returns `(low, high)` floats; `display_tier(clusterer, cluster_label_map, cluster_stats, bedrooms, area_m2, low, high)` returns a formatted string. Both functions are reused by the API via a new `predict_tier()` helper.
- `ml/prepare.py` — `ALL_FEATURES = ['bedrooms', 'area_m2', 'sector', 'property_type']`; column order matters when constructing the features DataFrame for the encoder.
- `chatbot/chat.py:main()` — shows the exact artifact loading pattern: 5 keys (`model`, `encoder`, `clusterer`, `cluster_label_map`, `cluster_stats`).
- `tests/test_chat.py` — mock pattern: `MagicMock` for model/encoder/clusterer; no conftest.py; plain `def test_*` functions; `assert` statements only.

### Institutional Learnings

- No prior API deployment learnings in `docs/solutions/`.

### External References

- **FastAPI lifespan pattern** (official docs): Use `@asynccontextmanager` passed to `FastAPI(lifespan=...)` — `@app.on_event("startup")` is deprecated since FastAPI 0.93.
- **TestClient usage**: Must use `with TestClient(app) as client:` (context manager form) when the app has a lifespan — without it, the lifespan does not execute and the model dict is empty.
- **Supervisor `stopasgroup=true` + `killasgroup=true`**: Required to kill Uvicorn worker child processes cleanly on restart; without them, workers become orphans.
- **`uvicorn[standard]`**: Installs `uvloop` and `httptools` for better performance on Linux — prefer over plain `uvicorn` in requirements.

## Key Technical Decisions

- **Lifespan context manager for model loading**: `@app.on_event` is deprecated; the lifespan pattern is the current FastAPI standard. The model dict is module-level so route handlers can access it without dependency injection complexity. (see origin)
- **New `predict_tier()` function in `chatbot/chat.py`**: `display_tier()` returns a formatted CLI string and is not touched. `predict_tier()` returns a structured dict `{market_tier, tier_stats}` with the same data. This avoids fragile string parsing and keeps the CLI contract unchanged. (see origin)
- **Absolute model path in `api.py`**: `Path(__file__).parent / 'ml' / 'model.pkl'` resolves relative to `api.py`'s location, not the process working directory. This prevents `FileNotFoundError` when Supervisor starts Uvicorn from a different working directory. (see origin)
- **Supervisor over systemd**: Simpler config for a learning project; `stopasgroup=true` and `killasgroup=true` prevent orphan worker processes on restart.
- **Pydantic validators for R2 constraints**: `bedrooms: int` with `ge=1` and `area_m2: float` with `gt=0` enforce R2 bounds; FastAPI returns HTTP 422 automatically — no manual validation code needed.

## Open Questions

### Resolved During Planning

- **Supervisor vs systemd?** Supervisor — simpler for a personal project; systemd adds no meaningful benefit at this scale.
- **`predict_tier()` location?** In `chatbot/chat.py` alongside `predict_range()` and `display_tier()` — keeps all prediction logic in one place; `api.py` imports from there.
- **Workers count for Uvicorn on t3.small?** 2 workers (t3.small has 2 vCPUs); formula `2 * cores + 1 = 3` is for high-concurrency services; 2 is sufficient for a demo/learning project.

### Deferred to Implementation

- **Exact Pydantic field validator syntax**: `Field(ge=1)` vs `Annotated[int, Field(ge=1)]` — either works; implementer should use whichever matches the installed Pydantic v2 convention.
- **Model path in tests**: Tests mock `joblib.load` so the real `.pkl` is never needed during CI; exact mock target (`api.joblib.load` vs patching the module-level dict) resolved during implementation.

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

```
Request flow:

  POST /predict  ─────────────────────────────────────────────────────────
  {sector, property_type, bedrooms, area_m2}
        │
        ▼ Pydantic validation (automatic HTTP 422 if invalid)
        │
        ▼ Normalize: sector.title(), property_type.lower()
        │
        ▼ pd.DataFrame([...], columns=ALL_FEATURES)
        │
        ▼ predict_range(model, encoder, features_df)  → (low, high)
        │
        ▼ predict_tier(clusterer, cluster_label_map,
                       cluster_stats, bedrooms, area_m2, low, high)
                                                       → {market_tier, tier_stats}
        │
        ▼ PredictResponse(price_low, price_high, market_tier, tier_stats)
        │
  HTTP 200  ──────────────────────────────────────────────────────────────

Startup (lifespan):

  joblib.load(MODEL_PATH)  →  model_store["model"],
                               model_store["encoder"],
                               model_store["clusterer"],
                               model_store["cluster_label_map"],
                               model_store["cluster_stats"]
```

## Implementation Units

- [ ] **Unit 1: Add `predict_tier()` to `chatbot/chat.py`**

  **Goal:** Extract the structured prediction data from the clustering logic into a new function that returns a dict instead of a formatted string — enabling the API to build its response without parsing the CLI output.

  **Requirements:** R3 (prerequisite)

  **Dependencies:** None

  **Files:**
  - Modify: `chatbot/chat.py`
  - Test: `tests/test_chat.py`

  **Approach:**
  - Add `predict_tier(clusterer, cluster_label_map, cluster_stats, bedrooms, area_m2, low, high)` that computes the midpoint, calls `clusterer.predict(...)`, resolves the label, and returns `{"market_tier": tier, "tier_stats": stats_dict}` where `stats_dict` is the raw dict from `cluster_stats[tier]`.
  - `display_tier()` is not changed — it can call `predict_tier()` internally and format the result, or keep its current logic. Either is fine; the important thing is that `predict_tier()` exists as a standalone testable function.
  - The new function follows the same DataFrame-input pattern as `display_tier()` for the clusterer predict call: the clusterer was trained on `['bedrooms', 'area_m2', 'price']` (in that order), so `predict()` must receive a DataFrame with exactly those three columns. Using a plain list instead of a named DataFrame triggers the sklearn feature-names warning fixed earlier.

  **Execution note:** Start with a failing test before adding the function.

  **Patterns to follow:**
  - `chatbot/chat.py:display_tier()` — same signature shape, same clusterer.predict call
  - `tests/test_chat.py:test_display_tier_*` — mock pattern with `_make_mock_clusterer()`

  **Test scenarios:**
  - Happy path: `predict_tier()` returns a dict with keys `market_tier` and `tier_stats`
  - Happy path: `market_tier` is one of `"Budget"`, `"Mid-Range"`, `"Luxury"`
  - Happy path: `tier_stats` contains `price_p10`, `price_p90`, `area_p10`, `area_p90`, `beds_p10`, `beds_p90`
  - Edge case: midpoint `(low + high) / 2` is correctly computed and passed to `clusterer.predict`

  **Verification:**
  - `tests/test_chat.py` passes including new `predict_tier` tests
  - `display_tier()` behavior is unchanged (existing tests still pass)

---

- [ ] **Unit 2: Create `api.py` with `POST /predict` and `GET /health`**

  **Goal:** Build the FastAPI application with the two required endpoints, loading the model once at startup via the lifespan pattern.

  **Requirements:** R1, R2, R3, R4, R5, R6, R9, R10, R11

  **Dependencies:** Unit 1

  **Files:**
  - Create: `api.py`
  - Test: `tests/test_api.py`

  **Approach:**
  - Define a module-level `model_store = {}` dict and populate it in the lifespan context manager using `joblib.load(MODEL_PATH)` where `MODEL_PATH = Path(__file__).parent / 'ml' / 'model.pkl'`.
  - Define `PredictRequest` (Pydantic BaseModel) with fields matching R2, using field validators for `bedrooms >= 1` and `area_m2 > 0`. FastAPI returns HTTP 422 automatically when validation fails.
  - Define `TierStats` and `PredictResponse` Pydantic models matching R3 structure. `TierStats` field names must mirror the artifact's dict keys **verbatim**: `price_p10`, `price_p90`, `area_p10`, `area_p90`, `beds_p10`, `beds_p90`. Using any other naming convention will produce incorrect JSON keys in the response without raising an error.
  - `POST /predict` handler: normalize inputs (R11), build `pd.DataFrame([...], columns=ALL_FEATURES)`, call `predict_range()`, call `predict_tier()`, return `PredictResponse`.
  - `GET /health` handler: return `{"status": "ok", "model_loaded": "model" in model_store}`.
  - Import `predict_range` and `predict_tier` from `chatbot.chat`; import `ALL_FEATURES` from `ml.prepare`.

  **Execution note:** Start with failing tests for both endpoints before writing `api.py`.

  **Patterns to follow:**
  - `chatbot/chat.py:main()` — model loading and prediction call pattern
  - FastAPI lifespan pattern from external research (asynccontextmanager, not @app.on_event)

  **Test scenarios:**
  - Happy path: `POST /predict` with valid JSON `{"sector": "Piantini", "property_type": "apartment", "bedrooms": 3, "area_m2": 120}` → HTTP 200 with `price_low`, `price_high`, `market_tier`, `tier_stats`
  - Happy path: `price_low <= price_high` in the response
  - Happy path: `market_tier` is one of `"Budget"`, `"Mid-Range"`, `"Luxury"`
  - Happy path: `GET /health` → HTTP 200 with `{"status": "ok", "model_loaded": true}`
  - Edge case: `sector` sent as `"piantini"` (lowercase) → response is valid (normalization applied)
  - Edge case: `property_type` sent as `"Apartment"` (uppercase) → response is valid (normalization applied)
  - Error path: missing required field `area_m2` → HTTP 422
  - Error path: `bedrooms: 0` → HTTP 422 (fails ge=1 constraint)
  - Error path: `area_m2: -10` → HTTP 422 (fails gt=0 constraint)
  - Error path: `property_type: "villa"` → HTTP 200 (unknown category handled gracefully by encoder's `handle_unknown='ignore'`; not a 422)
  - Integration: `TestClient` with `with TestClient(app) as client:` to trigger lifespan; mock `joblib.load` to avoid needing the real `.pkl` in tests

  **Verification:**
  - `tests/test_api.py` passes including validation and normalization tests
  - `uvicorn api:app --host 0.0.0.0 --port 8000` starts without error from the project root
  - `http://localhost:8000/docs` shows the Swagger UI with both endpoints

---

- [ ] **Unit 3: Update `requirements.txt` and add Supervisor config**

  **Goal:** Add the missing dependencies and provide the production deployment configuration for EC2.

  **Requirements:** R7, R8

  **Dependencies:** Unit 2

  **Files:**
  - Modify: `requirements.txt`
  - Create: `deploy/supervisor.conf`

  **Approach:**
  - Add `fastapi`, `uvicorn[standard]`, and `httpx` to `requirements.txt`. `httpx` is required by FastAPI's `TestClient` — without it, `from fastapi.testclient import TestClient` raises `ImportError` in a clean environment. `uvicorn[standard]` installs `uvloop` and `httptools` for better Linux performance.
  - Create `deploy/supervisor.conf` with:
    - `command` pointing to the venv's `uvicorn` binary with `api:app --host 0.0.0.0 --port 8000 --workers 2`
    - `directory=/home/ubuntu/Project1` (project root as working directory)
    - `autostart=true`, `autorestart=true`
    - `stopasgroup=true`, `killasgroup=true` (prevents orphan workers)
    - Log paths under `/var/log/predict-api/`
  - Include a short comment block at the top of the file explaining each critical setting.

  **Test expectation:** none — this unit contains only config files with no behavioral code.

  **Verification:**
  - `pip install -r requirements.txt` installs without error (including `fastapi` and `uvicorn`)
  - `deploy/supervisor.conf` exists with all required settings

## System-Wide Impact

- **`chatbot/chat.py` surface**: `predict_tier()` is additive — no existing function is changed. `display_tier()` and `predict_range()` are unchanged. The CLI continues to work identically.
- **`requirements.txt`**: Adding `fastapi` and `uvicorn[standard]` affects anyone who `pip install -r requirements.txt` — no conflict with existing deps (pandas, scikit-learn, joblib are all compatible).
- **Module import path**: `api.py` imports from `chatbot.chat` and `ml.prepare`. When run with `uvicorn api:app` from the project root, the root is on `sys.path` so these imports resolve. If run from a different directory, imports will fail — the Supervisor `directory=` directive handles this.
- **Unchanged invariants**: The CLI chatbot (`python chatbot/chat.py`), the training script (`python -m ml.train`), and all existing tests are completely unaffected by this change.

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| `import chatbot.chat` fails when uvicorn starts from wrong directory | Supervisor `directory=` set to project root; absolute model path as backup |
| `ml/model.pkl` not present on EC2 | Document in deploy notes: copy artifact or re-run `python -m ml.train data/listings.csv` on the instance before starting |
| Port 8000 blocked by EC2 Security Group | Document: add inbound rule for port 8000, source 0.0.0.0/0 |
| Uvicorn workers crash silently | Supervisor `autorestart=true` + stderr log at `/var/log/predict-api/stderr.log` |
| `property_type` values other than `apartment`/`house` return silent wrong predictions | Accepted — `handle_unknown='ignore'` in the encoder zeros out unknown categories; documented in API behavior |

## Documentation / Operational Notes

**EC2 setup sequence (for reference in deploy notes):**
1. Launch t3.small Ubuntu 22.04, open port 8000 in Security Group
2. SSH in, install Python deps: `sudo apt install python3-pip python3-venv git -y`
3. Clone/copy project, create venv, `pip install -r requirements.txt`
4. Verify model: `ls ml/model.pkl` (copy from local or re-train)
5. Install Supervisor: `sudo apt install supervisor -y`
6. Copy `deploy/supervisor.conf` to `/etc/supervisor/conf.d/predict-api.conf`
7. Create log dir: `sudo mkdir -p /var/log/predict-api`
8. `sudo supervisorctl reread && sudo supervisorctl update && sudo supervisorctl start predict-api`
9. Verify: `curl http://localhost:8000/health`

## Sources & References

- **Origin document:** [docs/brainstorms/2026-04-01-fastapi-endpoint-requirements.md](docs/brainstorms/2026-04-01-fastapi-endpoint-requirements.md)
- Related code: `chatbot/chat.py`, `ml/prepare.py`, `requirements.txt`
- FastAPI lifespan docs: https://fastapi.tiangolo.com/advanced/events/
- FastAPI TestClient docs: https://fastapi.tiangolo.com/tutorial/testing/
- Supervisor docs: http://supervisord.org/configuration.html
