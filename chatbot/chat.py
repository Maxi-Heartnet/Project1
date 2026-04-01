import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd

from ml.prepare import ALL_FEATURES

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


def display_tier(clusterer, cluster_label_map, cluster_stats, bedrooms, area_m2, low, high):
    midpoint = (low + high) / 2
    cluster_idx = clusterer.predict([[bedrooms, area_m2, midpoint]])[0]
    tier = cluster_label_map[cluster_idx]
    stats = cluster_stats[tier]
    price_lo = stats['price_p10'] / 1000
    price_hi = stats['price_p90'] / 1000
    area_lo = int(stats['area_p10'])
    area_hi = int(stats['area_p90'])
    beds_lo = int(stats['beds_p10'])
    beds_hi = int(stats['beds_p90'])
    return (
        f"Market tier: {tier}\n"
        f"  Typical in this tier: ${price_lo:.0f}k\u2013${price_hi:.0f}k USD"
        f" \u00b7 {area_lo}\u2013{area_hi} m\u00b2"
        f" \u00b7 {beds_lo}\u2013{beds_hi} bedrooms"
    )


def collect_features():
    prop_type = ask_string("Property type (apartment/house)? > ", choices=['apartment', 'house'])
    sector = ask_string("Sector (e.g. Piantini, Naco, Bella Vista)? > ")
    bedrooms = ask_number("Bedrooms? > ", dtype=int, min_val=1)
    area_m2 = ask_number("Area in m²? > ", dtype=float, min_val=1)
    return {
        'bedrooms': bedrooms,
        'area_m2': area_m2,
        'sector': sector.title(),
        'property_type': prop_type,
    }


def main(model_path=MODEL_PATH):
    artifact = joblib.load(model_path)
    model = artifact['model']
    encoder = artifact['encoder']
    clusterer = artifact['clusterer']
    cluster_label_map = artifact['cluster_label_map']
    cluster_stats = artifact['cluster_stats']

    print("\nWelcome! I'll estimate a house price in Santo Domingo.")
    print("Type 'exit' at any prompt to quit.\n")

    while True:
        features = collect_features()
        low, high = predict_range(model, encoder, pd.DataFrame([features], columns=ALL_FEATURES))
        print(f"\nEstimated price: ${low:,.0f} \u2013 ${high:,.0f} USD\n")
        tier_output = display_tier(
            clusterer, cluster_label_map, cluster_stats,
            bedrooms=features['bedrooms'],
            area_m2=features['area_m2'],
            low=low, high=high,
        )
        print(tier_output)
        print()
        again = ask_string("Estimate another? (yes/no) > ", choices=['yes', 'no'])
        if again == 'no':
            print("Goodbye!")
            break


if __name__ == '__main__':
    main()
