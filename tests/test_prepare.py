import pandas as pd
import pytest
from ml.prepare import clean, load_and_prepare


def make_df(rows):
    defaults = {
        'bedrooms': 3, 'bathrooms': 2, 'area_m2': 100, 'parking': 1,
        'floor_level': None, 'sector': 'Piantini', 'property_type': 'apartment',
        'price': 150_000,
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def test_clean_drops_missing_price():
    df = make_df([{}, {'price': None}])
    result = clean(df)
    assert len(result) == 1


def test_clean_drops_missing_area():
    df = make_df([{}, {'area_m2': None}])
    result = clean(df)
    assert len(result) == 1


def test_clean_removes_price_outliers():
    df = make_df([{}, {'price': 5_000}, {'price': 6_000_000}])
    result = clean(df)
    assert len(result) == 1
    assert result.iloc[0]['price'] == 150_000


def test_clean_fills_floor_level_with_zero():
    df = make_df([{'floor_level': None}])
    result = clean(df)
    assert result.iloc[0]['floor_level'] == 0


def test_load_and_prepare_shapes(tmp_path):
    rows = [
        {'bedrooms': (i % 4) + 1, 'bathrooms': 2, 'area_m2': 80 + i * 10,
         'parking': 1, 'floor_level': i % 5, 'sector': 'Naco',
         'property_type': 'apartment', 'price': 100_000 + i * 10_000}
        for i in range(20)
    ]
    csv = tmp_path / "listings.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    X_train, X_test, y_train, y_test, preprocessor = load_and_prepare(str(csv))
    assert X_train.shape[0] == 16   # 80% of 20
    assert X_test.shape[0] == 4
    assert X_train.shape[1] == X_test.shape[1]
