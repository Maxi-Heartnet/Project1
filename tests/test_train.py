import random
import joblib
import numpy as np
import pandas as pd
import pytest
from ml.train import train


def make_csv(tmp_path, n=40):
    random.seed(42)
    rows = [
        {
            'bedrooms': random.randint(1, 5),
            'area_m2': random.randint(50, 300),
            'sector': random.choice(['Piantini', 'Naco', 'Bella Vista']),
            'property_type': random.choice(['apartment', 'house']),
            'price': random.randint(80_000, 500_000),
        }
        for _ in range(n)
    ]
    csv = tmp_path / "listings.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return str(csv)


def test_train_saves_artifact_with_required_keys(tmp_path):
    csv = make_csv(tmp_path)
    model_path = str(tmp_path / "model.pkl")
    train(csv, model_path=model_path)
    artifact = joblib.load(model_path)
    assert 'model' in artifact
    assert 'encoder' in artifact
    assert 'clusterer' in artifact
    assert 'cluster_stats' in artifact
    assert 'cluster_label_map' in artifact


def test_cluster_stats_structure(tmp_path):
    csv = make_csv(tmp_path)
    model_path = str(tmp_path / "model.pkl")
    train(csv, model_path=model_path)
    artifact = joblib.load(model_path)
    cluster_stats = artifact['cluster_stats']
    assert set(cluster_stats.keys()) == {'Budget', 'Mid-Range', 'Luxury'}
    for label, stats in cluster_stats.items():
        for key in ('price_p10', 'price_p90', 'area_p10', 'area_p90', 'beds_p10', 'beds_p90'):
            assert key in stats, f"Missing key {key} in {label}"
        assert stats['price_p10'] <= stats['price_p90']
        assert stats['area_p10'] <= stats['area_p90']


def test_cluster_label_map_structure(tmp_path):
    csv = make_csv(tmp_path)
    model_path = str(tmp_path / "model.pkl")
    train(csv, model_path=model_path)
    artifact = joblib.load(model_path)
    label_map = artifact['cluster_label_map']
    assert len(label_map) == 3
    assert set(label_map.values()) == {'Budget', 'Mid-Range', 'Luxury'}


def test_cluster_labels_ordered_by_price(tmp_path):
    csv = make_csv(tmp_path)
    model_path = str(tmp_path / "model.pkl")
    train(csv, model_path=model_path)
    artifact = joblib.load(model_path)
    stats = artifact['cluster_stats']
    budget_mid = (stats['Budget']['price_p10'] + stats['Budget']['price_p90']) / 2
    midrange_mid = (stats['Mid-Range']['price_p10'] + stats['Mid-Range']['price_p90']) / 2
    luxury_mid = (stats['Luxury']['price_p10'] + stats['Luxury']['price_p90']) / 2
    assert budget_mid < midrange_mid < luxury_mid


def test_train_model_produces_positive_predictions(tmp_path):
    csv = make_csv(tmp_path)
    model_path = str(tmp_path / "model.pkl")
    train(csv, model_path=model_path)
    artifact = joblib.load(model_path)
    model = artifact['model']
    encoder = artifact['encoder']
    sample = pd.DataFrame([{
        'bedrooms': 3, 'area_m2': 120,
        'sector': 'Piantini', 'property_type': 'apartment',
    }])
    pred = model.predict(encoder.transform(sample))
    assert pred[0] > 0
