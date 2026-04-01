---
date: 2026-04-01
topic: market-segmentation
---

# Market Segmentation via Clustering

## Problem Frame

After the chatbot returns a price estimate, users have no context about where their property sits in the broader Santo Domingo market. Adding a market tier label (Budget / Mid-Range / Luxury) with supporting stats gives the price estimate meaning — users understand not just the number but the segment it belongs to.

This also introduces unsupervised learning (clustering) to the project alongside the existing supervised regression model, demonstrating how both paradigms can work together in one pipeline.

## Requirements

**Clustering model**
- R1. A K-Means clustering model (3 clusters) is trained on `[bedrooms, area_m2, price]` from the cleaned training data — raw numeric values only, not one-hot encoded. This keeps clusters meaningful, stable, and price-driven.
- R2. Clusters are auto-labeled by mean price: lowest mean = Budget, middle = Mid-Range, highest = Luxury.
- R3. The cluster model and pre-computed per-cluster stats are saved into the existing model artifact (`ml/model.pkl`) as new keys alongside `model` and `encoder`. Re-training the model is required before using the updated chatbot; no backward compatibility guard is added.

**Chatbot output**
- R4. After displaying the price estimate, the chatbot shows the tier the described property belongs to. Cluster assignment uses `[bedrooms, area_m2, midpoint_of_estimated_range]` as input to the cluster model.
- R5. The tier display includes a one-line stat summary: typical price range (10th–90th percentile of prices in that cluster), typical area range, and typical bedroom count range for that cluster. Stats are pre-computed at train time from the cleaned training data.

**Example output:**
```
Estimated price: $140,000 – $180,000 USD

Market tier: Mid-Range
  Typical in this tier: $110k–$210k USD · 70–130 m² · 2–3 bedrooms
```

## Success Criteria
- A user running the chatbot sees their tier and a one-line stat summary after every price estimate.
- The tier assignment uses the same feature encoding as the regressor — no second encoding step.
- Re-training the model updates both the regressor and the cluster labels without manual intervention.

## Scope Boundaries
- No standalone cluster-analysis mode or separate script.
- Number of tiers is fixed at 3 — not user-configurable.
- Tier labels are fixed (Budget / Mid-Range / Luxury) — not user-defined.
- No changes to the scraper or data pipeline.

## Key Decisions
- **3 clusters fixed**: Matches the natural Budget/Mid-Range/Luxury framing for real estate; interpretable without domain knowledge.
- **Cluster on raw numeric [bedrooms, area_m2, price]**: Avoids sparse 79-column one-hot matrix from one-hot encoding; K-Means on a 386×3 matrix is numerically stable. Price is the primary dimension for "Budget/Mid-Range/Luxury" classification.
- **Auto-label by mean price**: Removes any need for manual labeling after re-training on fresh data.
- **Stats pre-computed at train time**: The raw CSV is not available at chatbot load time. Stats are computed from the cleaned training DataFrame and stored in the artifact.
- **Stats from cluster members, not cluster centroid**: Percentile ranges from actual listings are more meaningful than centroid coordinates.
- **Require re-train, no backward compat guard**: This is a personal learning project. The plan will include a re-train step; chatbot code will not add defensive `.get()` fallbacks.
- **Saved in existing artifact**: Keeps the model artifact a single file with no new load paths in the chatbot.

## Dependencies / Assumptions
- The existing `ml/model.pkl` artifact dict will be extended with `clusterer` and `cluster_stats` keys.
- `load_and_prepare()` must be modified to also return the cleaned DataFrame (or its price/feature columns) so `train.py` can compute cluster stats before saving.

## Next Steps
→ `/ce:plan` for structured implementation planning
