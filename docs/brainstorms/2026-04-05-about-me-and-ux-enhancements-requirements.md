---
date: 2026-04-05
topic: about-me-and-ux-enhancements
---

# About Me Page & Form UX Enhancements

## Problem Frame

The prediction app currently has no identity — visitors have no way to know who built it or why.
The form also lacks basic UX conveniences: there is no way to reset it after getting an estimate,
and no shortcut for exploring the model with different inputs. These changes make the app more
compelling as a portfolio piece and more usable as a tool.

## Requirements

**About Me Page**

- R1. A new page is accessible at `GET /about`, served as an `HTMLResponse` by the existing
  FastAPI app (same delivery pattern as `GET /`).
- R2. The page presents a professional profile of Misael using the information established in this
  session. Content to include:
  - **Header**: Name — "Misael" (rendered as `<h2>`), title — "Full Stack & ML Developer"
    (rendered as a subtitle paragraph below the name)
  - **Summary**: Builder of end-to-end solutions spanning data science, backend APIs, web
    frontends, and cloud infrastructure; comfortable working across the full stack and deploying
    to production.
  - **Skills** (rendered as an `<h3>` section heading, categories as bold labels with bullet lists):
    - *Languages & Markup*: Python, JavaScript, HTML/CSS, HCL (Terraform)
    - *ML & Data*: scikit-learn, pandas, numpy, joblib — model training, pipelines,
      ColumnTransformer, serialization
    - *Backend*: FastAPI, uvicorn, REST API design
    - *Frontend*: Vanilla JS, async fetch, form validation, responsive layouts
    - *Cloud & Infra*: AWS EC2, Terraform (IaC), Nginx, Supervisor
    - *Tooling*: Git/GitHub, conventional commits, PR workflow, Claude Code (AI-assisted dev)
  - **Featured Project** (rendered as an `<h3>` section heading): Santo Domingo House Price Predictor — end-to-end ML application:
    trained a Random Forest on Dominican Republic real estate data, exposed it via a FastAPI
    REST endpoint, built a self-contained prediction form, and deployed the whole stack to AWS
    EC2 with a single `terraform apply`.
- R3. The About Me page includes a "Back to App" button that navigates to `GET /`.
- R4. The About Me page visual style matches the existing app (same color palette, font, card
  aesthetic) so it feels like one cohesive product.

**Main Page Enhancements**

- R5. The main prediction page includes a clearly labeled "About Me" button or link that
  navigates to `/about`. It is visible without scrolling (placed in the header or near the
  page title).
- R6. A "Clear" button is always visible in the button row (not conditional on a result being
  shown). Clicking it resets all four form fields to empty and hides the result panel if one
  is visible, returning the form to its initial state. The Clear and Fill randomly buttons are
  disabled while a prediction fetch is in-flight (i.e., while the Estimate Price button is
  disabled) and re-enabled when the fetch completes.
- R7. A "Fill randomly" button populates all four fields with valid random values drawn from
  available options:
  - **Sector**: randomly chosen from the `SECTORS` array already loaded in the page JS.
  - **Property type**: randomly chosen between "apartment" and "house".
  - **Bedrooms**: random integer between 1 and 5 (inclusive).
  - **Area m²**: random number between 50 and 400 m², rounded to one decimal place.
  After filling, any existing field-level validation errors are cleared and the result panel
  is hidden if one is currently visible.

## Success Criteria

- Visiting `/about` shows a polished, readable profile page that would make sense to a potential
  employer or collaborator seeing it cold.
- The "About Me" button is visible on the main page without scrolling.
- Clicking "Clear" on the main page empties all fields and removes the result panel in one click.
- Clicking "Fill randomly" populates all fields with plausible values; submitting the filled form
  returns a valid estimate without validation errors.
- The About Me page visually matches the main page (same header style, colors, typography).
- Pressing Space or Enter while the Clear or Fill randomly button is focused triggers the button
  action without submitting the form.
- While a prediction fetch is in-flight, Clear and Fill randomly are visually disabled and
  non-interactive.

## Scope Boundaries

- No authentication, no CMS, no database — the About Me content is hardcoded in the HTML
  template, the same way the main page is today.
- No contact form or mailto links unless trivially added.
- No new pip dependencies — CSS and JS move to static files served by FastAPI's built-in
  `StaticFiles` (already part of the framework; no additional packages required).
- The "Fill randomly" button does not auto-submit the form; it only populates fields.

## Key Decisions

- **Same delivery pattern**: About Me served as `HTMLResponse` from FastAPI, keeping the
  architecture simple and consistent with `GET /`.
- **Hardcoded content**: Profile text is embedded in the template, not fetched from a config
  or database. Simpler to maintain for a single-person portfolio project.
- **"Fill randomly" does not auto-submit**: Lets the user inspect the random values before
  committing, which is better for learning/exploration.
- **Static CSS and JS files**: CSS and JavaScript extracted from inline `_PAGE_HTML`/`_ABOUT_HTML`
  string constants into `static/style.css` and `static/main.js`, served via FastAPI's built-in
  `StaticFiles`. This reduces maintenance friction as the app grows and enables both pages to
  share the same stylesheet without duplicating styles.
- **Action buttons disabled during fetch**: Clear and Fill randomly are disabled while a
  prediction fetch is in-flight to prevent a race condition where the fetch result would
  overwrite a manually cleared or refilled form.

## Dependencies / Assumptions

- The `SECTORS` array is populated at request time via `__SECTORS__` placeholder substitution
  in `api.py`. With JS in a static file, the data bridge mechanism (e.g., a small inline
  `<script>` that sets `window.SECTORS = __SECTORS__;` before loading `main.js`) is a
  planning detail, not a product decision.
- Existing page styling (colors, fonts, card layout) is currently inline in `api.py`'s
  `_PAGE_HTML` constant. Moving it to `static/style.css` makes it the shared source of truth
  for both pages.

## Next Steps

→ `/ce:plan` for structured implementation planning
