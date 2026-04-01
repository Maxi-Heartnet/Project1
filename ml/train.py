import sys
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from ml.prepare import load_and_prepare

MODEL_PATH = 'ml/model.pkl'

TIER_LABELS = ['Budget', 'Mid-Range', 'Luxury']


def _build_cluster_artifacts(clusterer, df_train_clean):
    labels = clusterer.labels_
    mean_prices = {
        idx: df_train_clean.loc[df_train_clean.index[labels == idx], 'price'].mean()
        for idx in range(3)
    }
    sorted_indices = sorted(mean_prices, key=mean_prices.__getitem__)
    label_map = {idx: TIER_LABELS[rank] for rank, idx in enumerate(sorted_indices)}

    cluster_stats = {}
    for idx, tier in label_map.items():
        members = df_train_clean.iloc[labels == idx]
        cluster_stats[tier] = {
            'price_p10': float(np.percentile(members['price'], 10)),
            'price_p90': float(np.percentile(members['price'], 90)),
            'area_p10':  float(np.percentile(members['area_m2'], 10)),
            'area_p90':  float(np.percentile(members['area_m2'], 90)),
            'beds_p10':  float(np.percentile(members['bedrooms'], 10)),
            'beds_p90':  float(np.percentile(members['bedrooms'], 90)),
        }

    return label_map, cluster_stats


def train(csv_path, model_path=MODEL_PATH):
    X_train, X_test, y_train, y_test, preprocessor, df_train_clean = load_and_prepare(csv_path)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"MAE: ${mean_absolute_error(y_test, y_pred):,.0f}")
    print(f"R²:  {r2_score(y_test, y_pred):.3f}")

    clusterer = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusterer.fit(df_train_clean[['bedrooms', 'area_m2', 'price']])
    label_map, cluster_stats = _build_cluster_artifacts(clusterer, df_train_clean)

    joblib.dump({
        "model": model,
        "encoder": preprocessor,
        "clusterer": clusterer,
        "cluster_stats": cluster_stats,
        "cluster_label_map": label_map,
    }, model_path)
    print(f"Model saved to {model_path}")


if __name__ == '__main__':
    csv = sys.argv[1] if len(sys.argv) > 1 else 'data/listings.csv'
    train(csv)
