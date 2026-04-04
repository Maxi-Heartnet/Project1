---
title: "feat: Add web frontend for house price prediction"
type: feat
status: active
date: 2026-04-02
origin: docs/brainstorms/2026-04-02-frontend-requirements.md
---

# feat: Add web frontend for house price prediction

## Overview

Add a self-contained, single-page web UI served at `GET /` by the existing FastAPI app.
Anyone with the EC2 URL can fill in four property fields and receive a price range estimate
with a market tier badge — no curl, no Swagger, no Python required.

## Problem Frame

The prediction API is live but only accessible to technical users via Swagger or curl.
A general-public-facing HTML form removes that barrier without adding any new server,
build tool, or deployment step. The frontend ships as part of `api.py` on the same port
already open in the EC2 Security Group.
(see origin: docs/brainstorms/2026-04-02-frontend-requirements.md)

## Requirements Trace

- R1. Form with four fields: Sector (text + datalist), Property Type (dropdown: apartment/house), Bedrooms (int ≥ 1), Area m² (float > 0)
- R2. Client-side validation on blur and on submit; inline errors per field
- R3. Submit button labeled "Estimate Price"
- R4. Loading state: button disabled + label changes while request is in-flight
- R5. Price range headline: `$X,XXX – $X,XXX USD`
- R6. Market tier displayed as a colored badge (green/blue/gold)
- R7. Tier stats: typical price range, area range, bedroom range formatted as `$Xk – $Yk`, `X–Y m²`, `X–Y beds`
- R8. Results replace a placeholder area on the same page; no reload
- R9. Any non-2xx or network failure shows a user-friendly error in the results area; submit re-enables
- R10. Served by FastAPI at `GET /` — `HTMLResponse`, same port 8000
- R11. Self-contained HTML: inline CSS + inline JS, no CDN, no build step
- R12. Page title "Santo Domingo House Price Estimator"; readable on mobile (≥ 375px)
- R13. Badge colors: Budget = green, Mid-Range = blue, Luxury = gold/amber

## Scope Boundaries

- No HTTPS / custom domain — HTTP on port 8000 only (accepted risk for this phase)
- No rate limiting (accepted risk — POST /predict is public; documented below)
- No prediction history, user accounts, or persistent state
- No React, Vue, or other frontend framework — plain HTML/JS only
- No CDN dependencies — all JS and CSS inline in the HTML string
- No separate deployment step — ships as part of the existing FastAPI app
- Existing routes `/predict`, `/health`, `/docs` are unchanged
- `GET /` uses HTMLResponse (not StaticFiles) — avoids filesystem path dependency and route-shadowing risk
- `GET /` is a new route that replaces FastAPI's default root redirect, if any (intentional)

## Context & Research

### Relevant Code and Patterns

- `api.py:lifespan()` — existing pattern for model loading; extend to extract `known_sectors`
- `api.py:health()` — existing `@app.get` route; same decorator pattern for `GET /`
- `chatbot/chat.py:display_tier()` — canonical number format: prices as `$Xk` (÷1000, 0 decimals), area as `int(value)` m², beds as `int(value)` beds
- `chatbot/chat.py:collect_features()` — confirms property type choices are `['apartment', 'house']` and sector is free-text with examples "Piantini, Naco, Bella Vista"
- `ml/prepare.py:build_preprocessor()` — `OneHotEncoder(handle_unknown='ignore')` on `['sector', 'property_type']` in that order; sector categories are at index 0
- `tests/test_api.py:client fixture` — mock pattern: `patch('api.joblib')`, `MOCK_ARTIFACT` dict, `with TestClient(app) as c`

### Institutional Learnings

- No prior frontend or HTML-serving patterns in `docs/solutions/`.

### External References

- FastAPI `HTMLResponse`: returns `Content-Type: text/html`; import from `fastapi.responses`
- HTML5 `<datalist>`: native browser suggestion list for `<input>` fields; no JS library needed; degrades gracefully to plain text input when list is empty
- `JSON.stringify` / embedded JS variable: safe way to pass a Python list into an HTML template as a JS array literal

## Key Technical Decisions

- **`HTMLResponse` over `StaticFiles`**: A single inline HTML string requires no filesystem path, no static directory, and cannot accidentally shadow other routes. `StaticFiles` mounted at `/` must be added last to avoid shadowing `/predict` and `/health`. `HTMLResponse` has none of these constraints. (see origin)
- **Known sectors via `<datalist>` from encoder categories**: The fitted `OneHotEncoder` stores the sectors it was trained on in `.named_transformers_['cat'].categories_[0]`. Extracting these at startup and embedding as a JS array in the HTML gives users browser-native suggestions without autocomplete infrastructure or CDN dependencies. Graceful fallback: if extraction fails (e.g., encoder is not a ColumnTransformer or mock in tests), `known_sectors` defaults to `[]` and the field becomes a plain text input with placeholder text.
- **`fetch()` directly — no helper wrapper**: One fetch call on one page. A wrapper adds indirection with no second consumer. (see origin)
- **Display format matches `display_tier()`**: Tier stat prices are divided by 1000 and formatted as `$Xk`; area and beds are integers. Main price estimate uses comma-separated full numbers (e.g., `$145,000 – $182,000 USD`). This keeps the web UI consistent with the CLI chatbot.
- **Results area initial state is instructional copy**: "Fill in the form above to see a price estimate." — visible on load, replaced by results or error on first submit.
- **Re-submission clears results**: When the form is submitted while results are showing, the results area is cleared and replaced by a loading message until the new response arrives.
- **Validation on blur + submit**: Errors appear when a field loses focus or on submit attempt (whichever comes first). This is the standard convention — not on every keystroke.

## Open Questions

### Resolved During Planning

- **`HTMLResponse` vs `StaticFiles`?** → `HTMLResponse`. Lower complexity, no path dependency. (see Key Technical Decisions)
- **`fetch()` directly or helper wrapper?** → `fetch()` directly. (see Key Technical Decisions)
- **Exact tier stats display format?** → Follow `chatbot/chat.py:display_tier()`: `$Xk – $Yk USD`, `X–Y m²`, `X–Y beds`. (see Key Technical Decisions)
- **How to handle unknown sectors?** → HTML5 `<datalist>` sourced from encoder categories. If sector is unrecognized, prediction proceeds silently (encoder `handle_unknown='ignore'`); no server-side validation change needed.

### Deferred to Implementation

- **Exact encoder category extraction path**: `artifact['encoder'].named_transformers_['cat'].categories_[0].tolist()` — verify against the real artifact at implementation time. The lifespan should wrap this in try/except and default to `[]`.
- **MOCK_ARTIFACT encoder configuration for tests**: Update `_setup_mocks()` in `tests/test_api.py` to set `categories_` to numpy arrays: `MOCK_ARTIFACT['encoder'].named_transformers_['cat'].categories_ = [np.array([...sectors...]), np.array(['apartment', 'house'])]`. Plain Python lists have no `.tolist()` method and would silently trigger the try/except fallback, making all sector datalist tests fail.

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

```
GET / request flow:

  Browser GET /
        │
        ▼ FastAPI GET / handler
        │
        ▼ Read model_store['known_sectors'] (list from encoder)
        │
        ▼ Render HTML string (form + CSS + JS + sectors JSON embedded)
        │
  HTMLResponse(content=html_string, media_type="text/html")

HTML page interaction:

  Page load
     └─ Results area shows: "Fill in the form above to see a price estimate."

  User fills fields → blur validation (per-field inline errors)

  User clicks "Estimate Price"
     ├─ Client-side validation (all fields) → show errors if invalid
     └─ Valid → disable button, clear results, show "Getting estimate..."
                │
                ▼ fetch('POST /predict', JSON body)
                │
                ├─ 200 OK
                │    └─ render: price range headline
                │              + tier badge (colored by tier name)
                │              + tier stats (3 rows: price, area, beds)
                │
                └─ non-2xx or network error
                     └─ render: user-friendly error message
                                re-enable submit button
```

## Implementation Units

- [ ] **Unit 1: Extend lifespan and add `GET /` route to `api.py`**

  **Goal:** Serve the complete prediction web UI at `GET /` — a self-contained HTML page with form, inline CSS, inline JS, and a populated sector datalist. Also extend the lifespan to extract known sectors from the encoder and store in `model_store`.

  **Requirements:** R1–R13, R10 (serving)

  **Dependencies:** None (additive change to existing `api.py`)

  **Files:**
  - Modify: `api.py`

  **Approach:**
  - In `lifespan()`, after loading the artifact keys, extract known sectors from the encoder: `artifact['encoder'].named_transformers_['cat'].categories_[0].tolist()`. Wrap in try/except; default to `[]`. Store as `model_store['known_sectors']`.
  - Add `from fastapi.responses import HTMLResponse` import.
  - Add `@app.get('/', response_class=HTMLResponse)` route handler. It reads `model_store.get('known_sectors', [])`, JSON-serializes it, and embeds it in the HTML string.
  - The HTML string contains: `<head>` with title and inline `<style>`, `<body>` with form (4 fields + submit button) and a results `<div>`, and inline `<script>`.
  - Form fields: Sector `<input>` + `<datalist>` populated from the JS sectors variable; Property Type `<select>` with options `apartment`/`house` (HTML `value` attributes must remain lowercase to match API; display labels are "Apartment" and "House"); Bedrooms `<input type="number" min="1" step="1">`; Area m² `<input type="number" min="0.1" step="0.1">`. Each field has an associated `<label>` element.
  - CSS: minimal responsive layout. At ≥480px, form fields in a two-column grid. Below 480px, single column. Tier badge background colors: `#2e7d32` (Budget), `#1a6fbd` (Mid-Range), `#c9900a` (Luxury). Badge text color: white (`#ffffff`) for all tiers. Defined via CSS classes assigned by JS based on the `market_tier` value in the API response.
  - JS: `DOMContentLoaded` listener. Validation function per field (on blur + on submit). Sector: required, any text accepted (not validated against datalist). Bedrooms empty or < 1: show "Enter a whole number of bedrooms (1 or more)." Area empty or ≤ 0: show "Enter a positive area in m²." Sector empty: show "Please enter a sector or neighborhood." Fetch handler: POST to `/predict`, handle `response.ok`, parse JSON, render results. While in-flight: button label changes to "Estimating..." and results area shows "Getting your estimate...". Error handler for any non-2xx or network failure: show "Could not reach the server. Check your connection and try again." (network) or "Something went wrong. Please try again." (non-2xx server error) in results area; re-enable button and restore original label "Estimate Price".
  - Results rendering: headline `$X,XXX – $X,XXX USD`; tier badge; stats table with three rows using display_tier formatting (prices ÷ 1000 as `$Xk`).
  - Results area initial content: "Fill in the form above to see a price estimate."

  **Patterns to follow:**
  - `api.py:lifespan()` — existing key storage pattern for `model_store`
  - `api.py:health()` — `@app.get` decorator pattern
  - `chatbot/chat.py:display_tier()` — canonical number formatting (lines 66–76)

  **Test scenarios:**
  - *Test expectation:* This unit has no test file — tests are in Unit 2. Verify manually that `uvicorn api:app` starts, `curl http://localhost:8000/` returns HTML, and the browser form renders and submits successfully.

  **Verification:**
  - `GET /` returns HTTP 200 with `Content-Type: text/html`
  - The HTML response contains a `<form>`, a `<datalist>`, and a submit button labeled "Estimate Price"
  - The sector datalist is populated when the model is loaded (non-empty list on a real artifact)
  - Submitting the form in a browser renders a price range and a colored market tier badge

---

- [ ] **Unit 2: Tests for `GET /` in `tests/test_api.py`**

  **Goal:** Add test coverage for the new `GET /` route using the existing `client` fixture and mock patterns.

  **Requirements:** R1–R13 (observable via HTTP response)

  **Dependencies:** Unit 1

  **Files:**
  - Modify: `tests/test_api.py`

  **Approach:**
  - Update `_setup_mocks()` to configure the mock encoder with numpy arrays (not plain Python lists — plain lists have no `.tolist()` method and would trigger the try/except fallback silently): `MOCK_ARTIFACT['encoder'].named_transformers_['cat'].categories_ = [np.array(['Bella Vista', 'Naco', 'Piantini']), np.array(['apartment', 'house'])]`. `numpy` is already imported in `test_api.py`.
  - Add a `# --- Frontend route ---` section in the test file.
  - Tests cover: HTTP status, content-type, presence of key HTML elements, and that the sector datalist contains the known sectors.

  **Execution note:** Add tests before or alongside the implementation — both files will be changed in the same commit.

  **Patterns to follow:**
  - `tests/test_api.py:test_health_returns_200` — same `client` fixture, same assertion style
  - `tests/test_api.py:_setup_mocks()` — mock configuration pattern

  **Test scenarios:**
  - Happy path: `GET /` → HTTP 200
  - Happy path: `GET /` response has `Content-Type: text/html`
  - Happy path: response body contains `<form` (form element present)
  - Happy path: response body contains `"Estimate Price"` (submit button label)
  - Happy path: response body contains `<datalist` (sector suggestions element present)
  - Happy path: response body contains `"Piantini"` (a known sector from mock is embedded in the datalist)
  - Happy path: response body contains `"apartment"` and `"house"` (property type dropdown options)
  - Happy path: response body contains `"Santo Domingo House Price Estimator"` (page title per R12)

  **Verification:**
  - All new tests in `tests/test_api.py` pass
  - All pre-existing `test_api.py` tests still pass (no regression from mock changes)

## System-Wide Impact

- **Interaction graph:** `GET /` is a new read-only route. It reads `model_store` (populated by lifespan) and returns a static HTML string. No callbacks, middleware, or observers are triggered beyond normal FastAPI routing.
- **Error propagation:** If `model_store` is empty (lifespan failed), `model_store.get('known_sectors', [])` returns `[]` — the page still serves but the sector datalist is empty. This is acceptable graceful degradation.
- **State lifecycle risks:** `model_store['known_sectors']` is read-only after lifespan startup. No write path, no race condition, no cleanup concern. Two Uvicorn workers each have their own `model_store` copy — both load the same artifact, so both produce the same sector list.
- **API surface parity:** No existing API or CLI interface is changed. The CLI chatbot (`chatbot/chat.py`) is unaffected.
- **Integration coverage:** The `with TestClient(app) as c:` pattern in the existing test fixture fires the lifespan — `model_store['known_sectors']` will be populated during tests (from the mock encoder). The GET / tests verify this end-to-end.
- **Unchanged invariants:** `POST /predict`, `GET /health`, and `GET /docs` routes are completely unchanged. Existing tests continue to pass.

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Encoder category extraction fails in production (unexpected encoder structure) | Wrap in try/except; default `known_sectors = []`; sector field degrades to plain text input with placeholder |
| Rate limiting not in scope — `/predict` is public and unauthenticated | Accepted for this phase. Document in operational notes. Add `slowapi` rate limiting in a follow-up PR if abuse occurs |
| HTTP-only on port 8000 — MitM injection is theoretically possible | Accepted for this phase (matches existing API scope). Note: no sensitive data is collected (only property features, no PII) |
| `MOCK_ARTIFACT['encoder']` MagicMock needs specific attribute configuration for sector extraction | Resolved in Unit 2: update `_setup_mocks()` to configure `.named_transformers_['cat'].categories_` on the mock encoder |
| EC2 public IP is ephemeral — changes on instance restart | Not in scope. Document: attach an Elastic IP or use a DNS alias for shareable URLs |

## Documentation / Operational Notes

- After deploying, restart Supervisor: `sudo supervisorctl restart predict-api`
- Verify the frontend: `curl -I http://localhost:8000/` should return `Content-Type: text/html`
- Rate limiting note: `POST /predict` has no server-side rate limiting. If abuse is observed, add `slowapi` to `requirements.txt` and apply a per-IP limit (e.g., 10 req/min) on the `/predict` route. This does not affect the frontend route.
- The `/docs` Swagger UI remains accessible at `/docs` — consider restricting via security group or disabling in production (`docs_url=None` in FastAPI constructor) once the web UI is the primary interface.

## Sources & References

- **Origin document:** [docs/brainstorms/2026-04-02-frontend-requirements.md](docs/brainstorms/2026-04-02-frontend-requirements.md)
- Related code: `api.py`, `chatbot/chat.py` (display_tier), `tests/test_api.py`
- FastAPI HTMLResponse docs: https://fastapi.tiangolo.com/advanced/custom-response/
- HTML5 datalist: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/datalist
