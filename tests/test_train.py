import random
import joblib
import pandas as pd
import pytest
from ml.train import train


def make_csv(tmp_path, n=40):
    rows = [
        {
            'bedrooms': random.randint(1, 5),
            'bathrooms': random.randint(1, 3),
            'area_m2': random.randint(50, 300),
            'parking': random.randint(0, 3),
            'floor_level': random.randint(0, 10),
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


def test_train_model_produces_positive_predictions(tmp_path):
    csv = make_csv(tmp_path)
    model_path = str(tmp_path / "model.pkl")
    train(csv, model_path=model_path)
    artifact = joblib.load(model_path)
    model = artifact['model']
    encoder = artifact['encoder']
    sample = pd.DataFrame([{
        'bedrooms': 3, 'bathrooms': 2, 'area_m2': 120,
        'parking': 1, 'floor_level': 5,
        'sector': 'Piantini', 'property_type': 'apartment',
    }])
    pred = model.predict(encoder.transform(sample))
    assert pred[0] > 0
