---
title: "feat: Add market segmentation via K-Means clustering"
type: feat
status: completed
date: 2026-04-01
origin: docs/brainstorms/2026-04-01-market-segmentation-requirements.md
---

# feat: Add market segmentation via K-Means clustering

## Overview

After the chatbot shows a price estimate, it will also show the user which market tier their property belongs to (Budget / Mid-Range / Luxury) along with a one-line stat summary for that tier. A K-Means clustering model is trained alongside the existing RandomForest regressor and stored in the same artifact file.

## Problem Frame

Users get a price range but no market context — they don't know whether that price makes their property budget or high-end in Santo Domingo. Adding a tier label with stats turns the estimate from a number into meaningful market positioning. This also introduces unsupervised learning into the project alongside the existing supervised model.
(see origin: docs/brainstorms/2026-04-01-market-segmentation-requirements.md)

## Requirements Trace

- R1. K-Means (3 clusters) trained on raw `[bedrooms, area_m2, price]` from cleaned training data
- R2. Clusters auto-labeled by mean price: lowest = Budget, middle = Mid-Range, highest = Luxury
- R3. Clusterer and pre-computed per-cluster stats saved into existing `ml/model.pkl` artifact; re-train required before using updated chatbot
- R4. Chatbot assigns cluster using `[bedrooms, area_m2, midpoint_of_estimated_range]` and displays tier label after price estimate
- R5. Tier display includes one-line stat: price P10–P90, area P10–P90, bedroom P10–P90 for that cluster

## Scope Boundaries

- No standalone cluster analysis script or exploration mode
- Tier count and labels are fixed (3 tiers: Budget / Mid-Range / Luxury) — not configurable
- No changes to the scraper or CSV data pipeline
- No backward compatibility guard; old `ml/model.pkl` must be regenerated after this change

## Context & Research

### Relevant Code and Patterns

- `ml/prepare.py` — `load_and_prepare()` currently returns `(X_train, X_test, y_train, y_test, preprocessor)`. Must be extended to also return the cleaned DataFrame so `train.py` can derive cluster stats.
- `ml/train.py` — `train()` destructures the 5-tuple from `load_and_prepare()` and saves `{"model": ..., "encoder": ...}` via joblib. Will add clustering here.
- `chatbot/chat.py` — `main()` loads `artifact['model']` and `artifact['encoder']`. Will add `artifact['clusterer']` and `artifact['cluster_stats']` loads here. `predict_range()` returns `(low, high)` — midpoint is `(low + high) / 2`.
- `tests/test_prepare.py` — Destructures 5-tuple at line 43: `X_train, X_test, y_train, y_test, preprocessor = load_and_prepare(...)`. Will need updating for the new return value.
- `tests/test_train.py` — Checks `artifact` for `'model'` and `'encoder'` keys. Will add checks for `'clusterer'` and `'cluster_stats'`. The `make_csv()` fixture has stale columns (`bathrooms`, `parking`, `floor_level`) — harmless but worth cleaning up while touching this file.
- `sklearn.cluster.KMeans` — already available via `scikit-learn` in `requirements.txt`. No new dependency needed.

### Institutional Learnings

- No prior clustering learnings in `docs/solutions/`.

## Key Technical Decisions

- **Unscaled features for K-Means**: Price ($80k–$500k) dominates over bedrooms (1–5) and area (50–300m²) in Euclidean distance. This is intentional — Budget/Mid-Range/Luxury tiers should be price-ordered. Scaling would dilute the price signal and mix area-large-but-cheap properties with smaller expensive ones.
- **Cluster on training data only**: KMeans is fit on the 80% training split (≈309 rows). Stats are computed from the same training rows. Test set rows are not included in stats. This avoids data leakage while keeping the cluster model consistent with what the regressor was trained on.
- **`cluster_stats` as a dict keyed by tier label**: Structure `{"Budget": {price_p10, price_p90, area_p10, area_p90, beds_p10, beds_p90}, "Mid-Range": {...}, "Luxury": {...}}`. Serializes cleanly with joblib and is directly indexable by tier label in the chatbot.
- **Midpoint for cluster assignment in chatbot**: `(low + high) / 2` from `predict_range()` gives a single representative price to feed into `clusterer.predict([[bedrooms, area_m2, midpoint]])`.

## Open Questions

### Resolved During Planning

- **Should clustering use the full encoded feature matrix or raw numerics?** Raw numerics `[bedrooms, area_m2, price]` — decided in origin document. K-Means on a 386×3 matrix is stable; the 79-column one-hot matrix would be dominated by sparse binary sector dummies. (see origin)
- **Where do cluster stats come from?** Pre-computed at train time from the cleaned training DataFrame. The raw CSV is not available at chatbot load time. (see origin)
- **What happens if the model artifact is out of date?** No guard added — re-training is required. Stated in scope boundaries. (see origin)

### Deferred to Implementation

- **Exact K-Means `random_state` seed**: Any fixed seed works; implementer should choose one and verify that 3 clearly distinct clusters form on the real 386-row dataset. If cluster sizes are severely unbalanced (e.g., 1 row in a cluster), the implementer should try `n_init='auto'` or a different seed.
- **Formatting of stats output**: The requirements example uses `$110k–$210k USD · 70–130 m² · 2–3 bedrooms`. Exact rounding and unit formatting is up to the implementer, consistent with existing `${low:,.0f}` style in `chat.py`.

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

```
Training pipeline (ml/train.py):

  load_and_prepare(csv)
    → X_train, X_test, y_train, y_test, preprocessor, df_train_clean
                                                        ↑
                               cleaned DataFrame slice of the training rows
                               (same rows as X_train, before encoding)

  RandomForestRegressor.fit(X_train, y_train)         → artifact["model"]
  KMeans(n_clusters=3).fit(df_train_clean[features])  → artifact["clusterer"]
  auto_label(clusterer, df_train_clean)               → artifact["cluster_stats"]

  artifact["encoder"]        = preprocessor
  artifact["model"]          = regressor
  artifact["clusterer"]      = kmeans
  artifact["cluster_stats"]  = {label: {price_p10, ..., beds_p90}, ...}

Chatbot (chatbot/chat.py):

  predict_range(model, encoder, features_df)  → (low, high)
  midpoint = (low + high) / 2
  tier = LABELS[clusterer.predict([[bedrooms, area_m2, midpoint]])[0]]
  print tier + cluster_stats[tier]
```

## Implementation Units

- [ ] **Unit 1: Extend `load_and_prepare()` to return the cleaned training DataFrame**

  **Goal:** Make the cleaned training rows accessible to `train.py` for cluster fitting and stats computation without re-reading the CSV.

  **Requirements:** R1, R5 (prerequisite)

  **Dependencies:** None

  **Files:**
  - Modify: `ml/prepare.py`
  - Modify: `tests/test_prepare.py`

  **Approach:**
  - After the train/test split, return the portion of the cleaned DataFrame corresponding to the training indices alongside the existing 5 values: `(X_train, X_test, y_train, y_test, preprocessor, df_train_clean)`.
  - `df_train_clean` should be the raw (pre-encoded) rows: it only needs `bedrooms`, `area_m2`, and `price` for cluster fitting and stats, but returning the full slice is fine.
  - The 6-tuple return is a breaking change. All callers must be updated (currently just `ml/train.py` and `tests/test_prepare.py`).
  - Use `df.loc[X_train.index]` to reconstruct the training-set rows from the cleaned DataFrame. `X_train` from `train_test_split` carries the original pandas index, and `df[ALL_FEATURES]` shares that index. Do not use positional slicing here.

  **Execution note:** Start with a failing test before modifying `prepare.py`.

  **Patterns to follow:**
  - `tests/test_prepare.py:test_load_and_prepare_shapes` — existing shape test, update it to destructure 6 values and add an assertion on `df_train_clean`.

  **Test scenarios:**
  - Happy path: `load_and_prepare()` returns a 6-tuple; `df_train_clean` has 80% of input rows and contains `price`, `bedrooms`, `area_m2` columns
  - Edge case: `df_train_clean` row count matches `X_train` row count (no mismatch between encoded matrix and raw DataFrame)
  - Existing tests (`test_clean_drops_missing_price`, etc.) continue to pass unchanged

  **Verification:**
  - `test_prepare.py` passes with the 6-tuple destructure
  - `df_train_clean.shape[0]` equals `X_train.shape[0]`

---

- [ ] **Unit 2: Train K-Means and save clusterer + stats to artifact**

  **Goal:** Add unsupervised clustering to the training pipeline; extend the saved artifact with `clusterer` and `cluster_stats` keys.

  **Requirements:** R1, R2, R3

  **Dependencies:** Unit 1

  **Files:**
  - Modify: `ml/train.py`
  - Modify: `tests/test_train.py`

  **Approach:**
  - In `train()`, destructure the new 6-tuple from `load_and_prepare()`.
  - Fit `KMeans(n_clusters=3)` on `df_train_clean[['bedrooms', 'area_m2', 'price']]`.
  - Auto-label clusters: assign each cluster index a label by ranking cluster mean prices (lowest → Budget, middle → Mid-Range, highest → Luxury). The label mapping is a dict `{cluster_index: label}`.
  - For each label, compute stats from the training rows belonging to that cluster: `price_p10`, `price_p90`, `area_p10`, `area_p90`, `beds_p10`, `beds_p90` using the 10th and 90th percentiles of the raw training values.
  - Save `cluster_stats` as `{label: {stat_key: value, ...}}` — this is directly indexable in the chatbot.
  - Extend `joblib.dump` dict with `"clusterer"`, `"cluster_stats"`, and `"cluster_label_map"` keys.
  - **Fix broken existing test first**: `test_train_model_produces_positive_predictions` at `tests/test_train.py:43–48` passes a sample DataFrame with stale columns (`bathrooms`, `parking`, `floor_level`) to `encoder.transform()`. The real encoder is fitted only on `['bedrooms', 'area_m2', 'sector', 'property_type']` — extra columns cause a `ValueError`. Fix this before writing any clustering code: drop the stale columns from both the `make_csv()` fixture and the sample in `test_train_model_produces_positive_predictions`. This is a pre-existing bug, not a clustering change.
  - After fixing the baseline, update `test_train_saves_artifact_with_required_keys` to also assert `'clusterer'`, `'cluster_stats'`, and `'cluster_label_map'` are present.

  **Execution note:** Start with a failing test asserting the new artifact keys before modifying `train.py`.

  **Patterns to follow:**
  - `ml/train.py:train()` — existing artifact save pattern; extend the dict, don't replace it.
  - `tests/test_train.py:test_train_saves_artifact_with_required_keys` — mimic the existing key-presence assertion pattern for the two new keys.

  **Test scenarios:**
  - Happy path: artifact contains `'clusterer'`, `'cluster_stats'`, `'cluster_label_map'` after `train()` completes
  - Happy path: `cluster_stats` has exactly 3 keys: `'Budget'`, `'Mid-Range'`, `'Luxury'`
  - Happy path: `cluster_label_map` contains 3 entries mapping int indices 0/1/2 to label names
  - Happy path: each stats dict has keys `price_p10`, `price_p90`, `area_p10`, `area_p90`, `beds_p10`, `beds_p90`
  - Happy path: `price_p10 < price_p90` and `area_p10 < area_p90` for each cluster (percentiles are ordered)
  - Happy path: `Budget` mean price < `Mid-Range` mean price < `Luxury` mean price (label ordering is correct)
  - Edge case: fixture uses `n=40` (existing default). Add `random.seed(42)` at the top of `make_csv()` to eliminate flaky cluster-collapse risk from lopsided random price distributions across runs.

  **Verification:**
  - `test_train.py` passes with all new assertions
  - Running `python ml/train.py data/listings.csv` produces output without error and `ml/model.pkl` loads cleanly with all 4 keys

---

- [ ] **Unit 3: Display market tier in chatbot after price estimate**

  **Goal:** Load the cluster model and stats from the artifact, compute the tier for the user's property, and display it with a one-line stat summary after the price estimate.

  **Requirements:** R4, R5

  **Dependencies:** Unit 2 (re-trained artifact with `clusterer` and `cluster_stats` keys)

  **Files:**
  - Modify: `chatbot/chat.py`
  - Modify: `tests/test_chat.py`

  **Approach:**
  - In `main()`, extract `artifact['clusterer']` and `artifact['cluster_stats']` alongside `model` and `encoder`.
  - Pass `clusterer` and `cluster_stats` into a new `display_tier(clusterer, cluster_stats, bedrooms, area_m2, low, high)` helper (or inline in the loop — implementer's choice, but a helper makes it testable).
  - Inside the helper: compute `midpoint = (low + high) / 2`, call `clusterer.predict([[bedrooms, area_m2, midpoint]])[0]` to get the cluster index, map it to the label using the cluster model's internal label mapping, then look up `cluster_stats[label]` and format the output.
  - **Label mapping**: `KMeans.predict()` returns integer cluster indices (0, 1, 2). The artifact stores a `cluster_label_map = {int_index: label_name}` dict (computed in `train.py` and saved as the 5th artifact key alongside `model`, `encoder`, `clusterer`, `cluster_stats`). The chatbot loads it with `artifact['cluster_label_map']` and does `tier = cluster_label_map[clusterer.predict([[bedrooms, area_m2, midpoint]])[0]]`.
  - Format the stats line to match the example: `Typical in this tier: $Xk–$Yk USD · A–B m² · C–D bedrooms`.

  **Execution note:** Start with a failing test for `display_tier` before modifying `chat.py`.

  **Patterns to follow:**
  - `chatbot/chat.py:predict_range()` — existing testable helper pattern; follow the same signature style for `display_tier`.
  - `tests/test_chat.py` — existing mock-based test style using `MagicMock` for model/encoder; apply the same approach for clusterer mock.

  **Test scenarios:**
  - Happy path: `display_tier` returns a string containing the tier label (`Budget`, `Mid-Range`, or `Luxury`) when given valid inputs
  - Happy path: the returned string contains the stats values from `cluster_stats` for the matched tier
  - Integration: `collect_features()` + `predict_range()` + `display_tier()` chain produces non-empty output for a standard Piantini apartment input
  - Edge case: midpoint of a very high price range maps to `Luxury` tier; midpoint of a very low price range maps to `Budget` tier

  **Verification:**
  - `test_chat.py` passes including new tier tests
  - Running `python chatbot/chat.py` (after re-train) shows tier output after the price estimate matching the example format in the requirements

## System-Wide Impact

- **Artifact format change**: `ml/model.pkl` gains three new keys (`clusterer`, `cluster_stats`, `cluster_label_map`). Any code that loads the artifact (currently only `chatbot/chat.py`) must be updated. There are no other consumers.
- **`load_and_prepare()` return signature**: Changes from 5-tuple to 6-tuple. All callers updated in this plan (train.py, test_prepare.py). No other callers exist.
- **Re-train required**: The existing `ml/model.pkl` will not work with the updated `chat.py`. Users must run `python ml/train.py data/listings.csv` before running the chatbot.
- **Unchanged invariants**: The supervised regression path (`model`, `encoder`, `predict_range`) is not modified. The chatbot conversation flow, input prompts, and price estimate display are unchanged.

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| K-Means produces severely unbalanced clusters on 386 rows | Check cluster sizes in test output; if one cluster has <10 members, implementer should try a different `random_state` seed or inspect the data distribution |
| Label mapping lost between train and predict | Store `cluster_label_map = {int_index: label_name}` as a 5th artifact key; plan calls this out explicitly in Unit 3 |
| `test_train.py` fixture too small for 3 clusters | `make_csv(n=30)` gives ≥10 rows/cluster on average; sufficient for K-Means to converge |

## Sources & References

- **Origin document:** [docs/brainstorms/2026-04-01-market-segmentation-requirements.md](docs/brainstorms/2026-04-01-market-segmentation-requirements.md)
- Related code: `ml/prepare.py`, `ml/train.py`, `chatbot/chat.py`
- `sklearn.cluster.KMeans` — available via existing `scikit-learn` dependency
