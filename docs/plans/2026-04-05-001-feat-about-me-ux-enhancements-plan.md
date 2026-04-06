---
title: "feat: Add About Me page and form UX enhancements"
type: feat
status: active
date: 2026-04-05
origin: docs/brainstorms/2026-04-05-about-me-and-ux-enhancements-requirements.md
---

# feat: Add About Me page and form UX enhancements

## Overview

Deliver three things in sequence: (1) extract the existing inline CSS/JS from `_PAGE_HTML` into
`static/style.css` and `static/main.js` served by FastAPI's built-in `StaticFiles` mount; (2) add
the About Me navigation link, Clear button, and Fill randomly button to the main page with full JS
behavior; (3) add the `GET /about` profile page sharing the same stylesheet. The static-file
extraction is a prerequisite refactor — it eliminates the inline string maintenance problem and
enables both pages to share CSS without duplication.

## Problem Frame

The prediction app has no identity and the form lacks basic UX conveniences. The inline HTML/CSS/JS
string approach also creates growing maintenance friction. These changes address all three problems
at once: a shared static stylesheet, new interactive buttons, and a profile page. (see origin:
`docs/brainstorms/2026-04-05-about-me-and-ux-enhancements-requirements.md`)

## Requirements Trace

**About Me Page**
- R1. `GET /about` returns an `HTMLResponse` served by the existing FastAPI app
- R2. Profile: name "Misael" (`<h2>`), title "Full Stack & ML Developer" (subtitle `<p>`),
  summary, skills (`<h3>`) grouped by category with bullet lists, Featured Project (`<h3>`)
- R3. The About Me page includes a "Back to App" link navigating to `GET /`
- R4. The About Me page visual style matches the existing app (shared `static/style.css`)

**Main Page Enhancements**
- R5. An "About Me" link near the `<h1>`, visible without scrolling
- R6. A "Clear" button always visible; resets all fields, hides result panel, re-enables submit;
  disabled while a fetch is in-flight
- R7. A "Fill randomly" button populates all four fields with valid random values, clears errors,
  hides result panel; disabled while a fetch is in-flight

## Scope Boundaries

- No authentication, CMS, database, or contact form
- No new pip dependencies — `StaticFiles` is already part of FastAPI/starlette
- About Me content is hardcoded HTML, not fetched from a config or database
- "Fill randomly" does not auto-submit the form
- No changes to `/predict`, `/health`, or any `chatbot/` / `ml/` module
- `ml/prepare.py` sector normalization: no `.title()` applied — sectors stored as-is from source
  data; `api.py` uses `.strip()` only

## Context & Research

### Relevant Code and Patterns

- Current inline CSS/JS: `api.py` `_PAGE_HTML` lines 20–323 — all CSS in `<style>` block,
  all JS in IIFE; this is what Unit 0 extracts
- Route pattern: `api.py` lines 375–379 — synchronous `def`, `response_class=HTMLResponse`
- `StaticFiles` pattern: `from fastapi.staticfiles import StaticFiles`;
  `app.mount("/static", StaticFiles(directory="static"), name="static")` — no new packages
- SECTORS data bridge: `_PAGE_HTML` keeps a one-line inline `<script>` to inject
  `var SECTORS = __SECTORS__;` into global scope before `<script src="/static/main.js">` loads;
  `main.js` reads `SECTORS` from the global scope
- Color palette (from `api.py` inline CSS):
  - Background: `#f0f2f5`; Primary: `#1a6fbd`; Primary hover: `#155a9e`
  - Card: `#fff`, `border-radius: 10px`, `box-shadow: 0 1px 5px rgba(0,0,0,.09)`
  - Text: `#222` / `#444` / `#666`; Error: `#c0392b`
- Existing breakpoint: `@media (max-width: 479px)` in `_PAGE_HTML` — carry into `style.css`
- Test file: `tests/test_api.py` — `TestClient` with mocked `api.joblib`; `test_frontend_*`
  naming convention; `client` fixture handles mock setup — no changes needed
- `deploy/supervisor.conf` — uvicorn serves all routes on port 8000 directly; `static/` directory
  will be resolved relative to the working directory (`/home/ubuntu/Project1/`); no nginx or
  path routing changes needed

### Institutional Learnings

- No relevant learnings in `docs/solutions/` for this work

### External References

None needed — `StaticFiles` pattern is straightforward and well-documented within FastAPI.

## Key Technical Decisions

- **Static files via `StaticFiles` mount**: CSS and JS extracted to `static/style.css` and
  `static/main.js`. No build step, no bundler — plain CSS and ES5 JS served directly. Both pages
  link to the same stylesheet; the About Me page has no JS.
- **SECTORS data bridge**: `_PAGE_HTML` keeps a single inline `<script>var SECTORS = __SECTORS__;</script>`
  before the external script tag. `main.js` reads `SECTORS` from the global scope inside the IIFE.
  This is the only remaining inline script — everything else moves to `static/main.js`.
- **`_PAGE_HTML` and `_ABOUT_HTML` become HTML skeletons**: After Unit 0, these constants contain
  only the HTML structure with `<link>` and `<script>` tags. The About Me page has no `<script>`
  at all — its Back to App anchor is plain HTML.
- **Action buttons disabled during in-flight fetch (R6, R7)**: In the fetch submit handler,
  `clearBtn.disabled = true` and `fillBtn.disabled = true` are set alongside `btn.disabled = true`;
  both the `.then()` success path and the `.catch()` error path re-enable all three. This closes
  the race condition where a fetch resolving after Clear/Fill would overwrite the new state.
- **About Me heading hierarchy (R2)**: name as `<h2>`, title as `<p class="subtitle">`,
  section labels (Skills, Featured Project) as `<h3>`. Add `.subtitle` CSS rule to `style.css`.
- **About Me page shares `static/style.css`**: No per-page CSS duplication. The About Me page
  links to the same stylesheet; any class defined there (`.page-header`, `.btn-secondary`,
  `.card`, etc.) is available to both pages without copying.
- **`clearForm()` and `fillRandom()` re-enable submit button**: The existing fetch success handler
  does not re-enable the submit button; both new handlers must reset `btn.disabled = false` and
  `btn.textContent = 'Estimate Price'`.

## Open Questions

### Resolved During Planning

- **R6 Clear button always-visible**: Always visible, not conditional on result shown
- **R7 Fill randomly + result panel**: Hides result panel on click
- **Visual style operationalization**: Exact palette confirmed from `api.py` lines 27–108
- **Submit button re-enable**: Handled in both `clearForm()` and `fillRandom()`
- **Action buttons during in-flight fetch**: Disabled alongside submit button; re-enabled in
  both success and error paths
- **About Me heading hierarchy**: `<h2>` name, `<p class="subtitle">` title, `<h3>` sections
- **Static files**: CSS/JS extracted to `static/`; no new pip packages; SECTORS via global var
- **Sector normalization bug**: Fixed — `api.py` now uses `.strip()` instead of `.title()`

### Deferred to Implementation

- **Secondary button focus-visible style**: Add `outline: 2px solid #1a6fbd; outline-offset: 2px`
  to `.btn-secondary:focus-visible` in `style.css` to match the existing `input:focus` pattern
- **Skills section layout**: Grouped bullet lists (one `<ul>` per category under a bold label)
  is the simpler default; the `.form-grid` two-column pattern is available if preferred
- **Back to App navigation is stateless**: Full page reload — form fields will be empty on return.
  Expected and acceptable; no session storage needed.

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not
> implementation specification. The implementing agent should treat it as context, not
> code to reproduce.*

```
Project structure after all units:

static/
  style.css          ← extracted from _PAGE_HTML + new rules (.page-header, .btn-secondary,
                        .action-row, .subtitle, h2/h3 styles)
  main.js            ← extracted IIFE from _PAGE_HTML + new clearForm(), fillRandom(),
                        action button disable logic

api.py:
  _PAGE_HTML = """   ← HTML skeleton only; links to style.css; inline SECTORS script;
    <link href="/static/style.css">        links to main.js
    <script>var SECTORS = __SECTORS__;</script>
    <script src="/static/main.js"></script>
  """

  _ABOUT_HTML = """  ← HTML skeleton only; links to style.css; no JS
    <link href="/static/style.css">
  """

  app.mount("/static", StaticFiles(...))
  @app.get('/')      ← returns modified _PAGE_HTML (still does __SECTORS__ substitution)
  @app.get('/about') ← returns _ABOUT_HTML directly
  @app.get('/health')
  @app.post('/predict')
```

Page layout for `GET /` (HTML structure — CSS lives in style.css):

```
<header class="page-header">          ← R5
  <h1>Santo Domingo House Price...</h1>
  <a href="/about" class="btn-secondary">About Me</a>
</header>
<div class="card">
  <form id="predict-form">
    [form grid]
    <div class="submit-row">
      <button id="submit-btn" type="submit">Estimate Price</button>
    </div>
    <div class="action-row">           ← R6, R7
      <button id="clear-btn" type="button">Clear</button>
      <button id="fill-btn"  type="button">Fill randomly</button>
    </div>
  </form>                              ← form closes here
</div>                                 ← card closes here
<div id="results">                     ← outside the form
```

## Implementation Units

```mermaid
TB
  U0["Unit 0\nExtract static files"] --> U1["Unit 1\nMain page enhancements"]
  U0 --> U2["Unit 2\nAbout Me page"]
  U1 --> U2
```

- [ ] **Unit 0: Extract CSS and JS to static files**

**Goal:** Move all CSS and JS out of `_PAGE_HTML` into `static/style.css` and `static/main.js`.
After this unit, `_PAGE_HTML` is a slim HTML skeleton and all existing tests still pass.

**Requirements:** prerequisite for R4 (shared style), R6, R7

**Dependencies:** None

**Files:**
- Create: `static/style.css`
- Create: `static/main.js`
- Modify: `api.py`
- Test: `tests/test_api.py`

**Approach:**
- Create `static/` directory at project root
- Extract the full `<style>` block content (everything between `<style>` and `</style>` in
  `_PAGE_HTML`) into `static/style.css` verbatim; keep the existing `@media (max-width: 479px)`
  block
- Extract the full IIFE (everything between `<script>` and `</script>` in `_PAGE_HTML`) into
  `static/main.js` verbatim; the IIFE reads `SECTORS` from the global scope (it was declared
  with `var SECTORS = __SECTORS__` in the same inline script block — separate it into:
  `_PAGE_HTML` keeps `<script>var SECTORS = __SECTORS__;</script>`, `main.js` gets the IIFE)
- Replace the `<style>...</style>` block in `_PAGE_HTML` with
  `<link rel="stylesheet" href="/static/style.css">`
- Replace the `<script>...</script>` block in `_PAGE_HTML` with:
  `<script>var SECTORS = __SECTORS__;</script>` (data bridge) followed by
  `<script src="/static/main.js"></script>`
- Add `from fastapi.staticfiles import StaticFiles` import to `api.py`
- Mount static files: `app.mount("/static", StaticFiles(directory="static"), name="static")`
  — place after `app = FastAPI(...)` and before the route definitions

**Patterns to follow:**
- `api.py` lines 345–349 — `app = FastAPI(...)` declaration; mount goes immediately after
- Existing `__SECTORS__` replacement in `index()` — unchanged; still replaces the placeholder
  in the inline data bridge script

**Test scenarios:**
- Regression: `GET /` returns 200 with `text/html` content type
- Regression: `GET /` response body no longer contains `<style>` or inline `<script>` blocks
  (CSS and JS have moved)
- Regression: `GET /` response body contains `href="/static/style.css"`
- Regression: `GET /` response body contains `src="/static/main.js"`
- Regression: `GET /` response body still contains `__SECTORS__` placeholder absent and
  `'"Piantini"'` present (sector injection still works)
- Regression: all existing `test_frontend_*` tests continue to pass

**Verification:**
- All existing `test_frontend_*` tests pass with no changes
- In a browser: `GET /` renders identically to before — same styling, same form behavior,
  same sector autocomplete

---

- [ ] **Unit 1: Main page — header link, action buttons, and JS handlers**

**Goal:** Add the About Me navigation link (R5), Clear button (R6), and Fill randomly button (R7)
with all CSS in `static/style.css` and all JS in `static/main.js`.

**Requirements:** R5, R6, R7

**Dependencies:** Unit 0 — `static/style.css` and `static/main.js` must exist

**Files:**
- Modify: `static/style.css`
- Modify: `static/main.js`
- Modify: `api.py` (HTML structure in `_PAGE_HTML` only)
- Test: `tests/test_api.py`

**Approach:**

*CSS additions to `static/style.css`:*
- Add `.page-header`: `display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 1.25rem;` — replace `h1`'s `margin-bottom` with this wrapper's margin
- Add `.btn-secondary`: `border: 1px solid #1a6fbd; color: #1a6fbd; background: #fff;
  border-radius: 6px; padding: .5rem .9rem; font-size: .9rem; font-weight: 600; cursor: pointer;
  text-decoration: none; display: inline-block;`
- Add `.btn-secondary:hover`: `background: #e8f0f9`
- Add `.btn-secondary:focus-visible`: `outline: 2px solid #1a6fbd; outline-offset: 2px`
- Add `.btn-secondary:disabled`: `opacity: .6; cursor: not-allowed`
- Add `.action-row`: `display: flex; gap: .75rem; margin-top: .65rem; width: 100%;`
- Add `.action-row .btn-secondary`: `flex: 1; text-align: center`
- Inside existing `@media (max-width: 479px)` block: add `.page-header { flex-direction: column;
  align-items: flex-start; gap: .5rem; }`

*HTML changes to `_PAGE_HTML` in `api.py`:*
- Wrap `<h1>` and `<a href="/about" class="btn-secondary">About Me</a>` in
  `<header class="page-header">`
- Add `<div class="action-row">` after `.submit-row` (inside the form, before `</form>`)
  containing `<button type="button" id="clear-btn" class="btn-secondary">Clear</button>` and
  `<button type="button" id="fill-btn" class="btn-secondary">Fill randomly</button>`

*JS additions to `static/main.js` IIFE (after the `var btn` and `var form` declarations):*
- Add `var clearBtn = document.getElementById('clear-btn');`
- Add `var fillBtn = document.getElementById('fill-btn');`
- Add `clearForm()` function: iterates `fields` setting `.value = ''` and `setError(id, '')`,
  declares local `var r = document.getElementById('results')`, calls `clearResults()` then sets
  `r.textContent = 'Fill in the form above to see a price estimate.'`, resets `btn.disabled =
  false` and `btn.textContent = 'Estimate Price'`
- Add `fillRandom()` function: guards `if (!SECTORS.length) { return; }`, picks random sector
  (`SECTORS[Math.floor(Math.random() * SECTORS.length)]`), random type
  (`['apartment','house'][Math.floor(Math.random()*2)]`), random bedrooms
  (`Math.floor(Math.random()*5)+1`), random area
  (`Math.round((Math.random()*350+50)*10)/10`); sets each field's `.value`; calls
  `setError(f.id,'')` for each field; restores placeholder; resets btn
- Modify the existing fetch submit handler: add `clearBtn.disabled = true;
  fillBtn.disabled = true;` alongside `btn.disabled = true;`; in both `.then()` and `.catch()`
  paths add `clearBtn.disabled = false; fillBtn.disabled = false;`
- Wire click handlers: `clearBtn.addEventListener('click', clearForm);` and
  `fillBtn.addEventListener('click', fillRandom);` after the `form.addEventListener` block
- Maintain ES5-style `var`/`function` throughout

**Patterns to follow:**
- `static/main.js` IIFE structure (from Unit 0 extraction)
- Existing `setError` / `fields` array iteration pattern in `main.js`
- Existing `clearResults()` helper in `main.js`

**Test scenarios:**
- Happy path: `GET /` response body contains text `"Clear"`
- Happy path: `GET /` response body contains text `"Fill randomly"`
- Happy path: `GET /` response body contains `href="/about"` and text `"About Me"`
- Regression: `GET /` response body contains `id="predict-form"` (form id unchanged)
- Regression: `GET /` response body contains `id="submit-btn"` (button id unchanged)
- Keyboard: focusing `clear-btn` and pressing Enter triggers `clearForm()` without form
  submission — verify by checking that the submit handler's `e.preventDefault()` path is not
  hit (type="button" prevents this; test by confirming no fetch is initiated)
- Keyboard: same for `fill-btn`
- Regression: all existing `test_frontend_*` tests pass

**Verification:**
- `GET /` returns 200 and HTML contains the About Me link, Clear button, and Fill randomly button
- In a browser: Clear resets fields, shows placeholder, re-enables Estimate Price
- In a browser: Fill randomly populates all four fields; submitting returns a valid prediction
- In a browser: clicking Clear or Fill randomly while a fetch is in-flight shows them disabled;
  they re-enable when the fetch completes
- About Me link navigates to `/about`

---

- [ ] **Unit 2: About Me page — `_ABOUT_HTML` and `GET /about` route**

**Goal:** Add a polished profile page at `GET /about` sharing `static/style.css` (R1–R4).

**Requirements:** R1, R2, R3, R4

**Dependencies:** Unit 0 (`static/style.css` exists); Unit 1 (`.page-header`, `.btn-secondary`,
`.subtitle`, `h2`/`h3` styles established in `style.css`)

**Files:**
- Modify: `static/style.css` (add `.subtitle`, `h2`, `h3` rules if not already present)
- Modify: `api.py`
- Test: `tests/test_api.py`

**Approach:**

*CSS additions to `static/style.css` (if not already added in Unit 1):*
- `h2`: suitable size and weight (e.g., `font-size: 1.4rem; font-weight: 700; color: #111;
  margin-bottom: .25rem;`)
- `h3`: section label style (e.g., `font-size: 1rem; font-weight: 700; color: #333;
  margin: 1rem 0 .4rem;`)
- `.subtitle`: muted subheading (e.g., `font-size: 1rem; color: #555; margin-bottom: 1rem;`)

*`_ABOUT_HTML` constant in `api.py`:*
- Add immediately after `_PAGE_HTML`, before `lifespan`
- Structure: doctype, viewport meta, `<link rel="stylesheet" href="/static/style.css">`,
  **no `<script>`** — the About Me page has no JavaScript
- `<main>` → `<header class="page-header">` with page title on left and
  `<a href="/" class="btn-secondary">← Back to App</a>` on right (R3)
- `<div class="card">` containing:
  - `<h2>Misael</h2>` and `<p class="subtitle">Full Stack &amp; ML Developer</p>`
  - Summary paragraph
  - `<h3>Skills</h3>` with six categories, each as a `<strong>` label + `<ul>`
  - `<h3>Featured Project</h3>` with Santo Domingo House Price Predictor description

*Route:*
- `@app.get('/about', response_class=HTMLResponse)` returning `HTMLResponse(content=_ABOUT_HTML)`
  — synchronous `def`, no `model_store` access, no placeholder substitution

**Patterns to follow:**
- `_PAGE_HTML` skeleton structure post-Unit-0 (doctype, viewport, link to style.css, main wrapper)
- `api.py` `index()` route definition
- `.page-header` / `.btn-secondary` usage established in Unit 1

**Test scenarios:**
- Happy path: `GET /about` returns 200
- Happy path: response `Content-Type` starts with `text/html`
- Happy path: response body contains `"Misael"`
- Happy path: response body contains `"Full Stack"`
- Happy path: response body contains `href="/"`
- Happy path: response body contains `"FastAPI"` (skills section present)
- Happy path: response body contains `"Santo Domingo House Price Predictor"`
- Happy path: response body contains `href="/static/style.css"` (links to shared stylesheet)
- Happy path: response body does NOT contain `<script` (no JS on About Me page)

**Verification:**
- All nine test scenarios pass
- In a browser: About page visually matches the main page — same card, colors, header pattern
- `← Back to App` navigates to `GET /`

## System-Wide Impact

- **Interaction graph:** `api.py`, `static/style.css`, `static/main.js` modified. No callbacks,
  middleware, or other modules affected. `chatbot/` and `ml/` unchanged.
- **Error propagation:** If `static/` directory is absent at startup, `StaticFiles` raises on
  mount — app fails to start. Mitigation: Unit 0 creates the directory before mounting.
  `GET /about` has no failure modes (static HTML, no `model_store` access).
- **State lifecycle risks:** `clearForm()` and `fillRandom()` re-enable the submit button,
  recovering from the latent state where a successful prediction leaves it disabled. Action
  buttons are now explicitly disabled during fetch to prevent race conditions.
- **API surface parity:** `/predict`, `/health`, and `GET /` contract unchanged.
- **Integration coverage:** Static file serving adds a new FastAPI mount; the `TestClient` in
  tests does not serve static files by default — CSS/JS links in the HTML will appear as
  unresolvable URLs in `TestClient` responses, but this does not affect any existing assertions
  (tests check HTML structure, not stylesheet application).
- **Unchanged invariants:** All `test_frontend_*`, `test_predict_*`, and `test_health_*` tests
  remain valid; `client` fixture requires no changes.

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| `static/` directory absent on EC2 after `git pull` — `StaticFiles` raises on startup | `static/` and its files are committed to the repo; they will be present after clone/pull |
| `TestClient` does not resolve `/static/style.css` or `/static/main.js` | Tests only assert HTML structure, not rendered appearance or script execution — no test breakage |
| `SECTORS` global var leaked into window scope | Acceptable for a single-page app with no third-party JS; risk is negligible |
| `_PAGE_HTML` string edit regression during Unit 0 | Run all `test_frontend_*` tests after Unit 0 extraction before proceeding to Unit 1 |
| Action button disable in `.then()` success path omitted | Add `clearBtn.disabled = false; fillBtn.disabled = false;` in success path alongside btn re-enable; submit handler currently only re-enables btn in catch — fix both paths in Unit 1 |

## Sources & References

- **Origin document:** [docs/brainstorms/2026-04-05-about-me-and-ux-enhancements-requirements.md](docs/brainstorms/2026-04-05-about-me-and-ux-enhancements-requirements.md)
- Related code: `api.py` — `_PAGE_HTML` (lines 20–323), `index()` (lines 375–379)
- Related code: `tests/test_api.py` — `test_frontend_*` test group, `client` fixture
