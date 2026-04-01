import pandas as pd
import pytest
from ml.prepare import clean, load_and_prepare


def make_df(rows):
    defaults = {
        'bedrooms': 3, 'area_m2': 100,
        'sector': 'Piantini', 'property_type': 'apartment',
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


def test_load_and_prepare_shapes(tmp_path):
    rows = [
        {'bedrooms': (i % 4) + 1, 'area_m2': 80 + i * 10,
         'sector': 'Naco',
         'property_type': 'apartment', 'price': 100_000 + i * 10_000}
        for i in range(20)
    ]
    csv = tmp_path / "listings.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    X_train, X_test, y_train, y_test, preprocessor, df_train_clean = load_and_prepare(str(csv))
    assert X_train.shape[0] == 16   # 80% of 20
    assert X_test.shape[0] == 4
    assert X_train.shape[1] == X_test.shape[1]
    assert df_train_clean.shape[0] == X_train.shape[0]
    assert {'price', 'bedrooms', 'area_m2'}.issubset(df_train_clean.columns)


def test_preprocessor_ignores_unknown_sector():
    from ml.prepare import build_preprocessor, ALL_FEATURES
    preprocessor = build_preprocessor()
    train_df = make_df([{}])  # only 'Piantini' seen during fit
    preprocessor.fit(train_df[ALL_FEATURES])
    test_df = make_df([{'sector': 'Los Cacicazgos'}])
    result = preprocessor.transform(test_df[ALL_FEATURES])
    assert result.shape[0] == 1  # did not raise, returned one row
