import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from chatbot.chat import ask_string, ask_number, collect_features, predict_range, display_tier, predict_tier


def test_ask_string_accepts_valid_choice():
    with patch('builtins.input', return_value='apartment'):
        assert ask_string("p > ", choices=['apartment', 'house']) == 'apartment'


def test_ask_string_rejects_then_accepts():
    with patch('builtins.input', side_effect=['villa', 'house']):
        assert ask_string("p > ", choices=['apartment', 'house']) == 'house'


def test_ask_string_no_choices_returns_value():
    with patch('builtins.input', return_value='Piantini'):
        assert ask_string("p > ") == 'Piantini'


def test_ask_number_valid_int():
    with patch('builtins.input', return_value='3'):
        assert ask_number("p > ", dtype=int, min_val=1) == 3


def test_ask_number_rejects_non_numeric_then_accepts():
    with patch('builtins.input', side_effect=['abc', '120']):
        assert ask_number("p > ", dtype=float, min_val=1) == 120.0


def test_ask_number_rejects_below_min():
    with patch('builtins.input', side_effect=['0', '1']):
        assert ask_number("p > ", dtype=int, min_val=1) == 1


def test_predict_range_low_lte_high():
    mock_tree = MagicMock()
    mock_tree.predict.return_value = np.array([150_000.0])
    mock_model = MagicMock()
    mock_model.estimators_ = [mock_tree] * 20

    mock_encoder = MagicMock()
    mock_encoder.transform.return_value = np.zeros((1, 8))

    features_df = pd.DataFrame([{
        'bedrooms': 3, 'area_m2': 120,
        'sector': 'Piantini', 'property_type': 'apartment',
    }])
    low, high = predict_range(mock_model, mock_encoder, features_df)
    assert low <= high
    assert low > 0


def test_collect_features_returns_correct_keys():
    with patch('builtins.input', side_effect=['apartment', 'Piantini', '3', '120']):
        features = collect_features()
    assert set(features.keys()) == {'bedrooms', 'area_m2', 'sector', 'property_type'}
    assert features['property_type'] == 'apartment'
    assert features['bedrooms'] == 3
    assert features['area_m2'] == 120.0
    assert features['sector'] == 'Piantini'


# --- display_tier tests ---

MOCK_CLUSTER_STATS = {
    'Budget':    {'price_p10': 80_000,  'price_p90': 150_000, 'area_p10': 50,  'area_p90': 100, 'beds_p10': 1, 'beds_p90': 2},
    'Mid-Range': {'price_p10': 150_000, 'price_p90': 280_000, 'area_p10': 90,  'area_p90': 160, 'beds_p10': 2, 'beds_p90': 3},
    'Luxury':    {'price_p10': 280_000, 'price_p90': 500_000, 'area_p10': 140, 'area_p90': 300, 'beds_p10': 3, 'beds_p90': 5},
}

MOCK_LABEL_MAP = {0: 'Budget', 1: 'Mid-Range', 2: 'Luxury'}


def _make_mock_clusterer(predicted_index):
    mock = MagicMock()
    mock.predict.return_value = np.array([predicted_index])
    return mock


def test_display_tier_returns_string_with_tier_label():
    clusterer = _make_mock_clusterer(1)  # Mid-Range
    result = display_tier(clusterer, MOCK_LABEL_MAP, MOCK_CLUSTER_STATS, bedrooms=3, area_m2=120, low=160_000, high=200_000)
    assert 'Mid-Range' in result


def test_display_tier_contains_stats_values():
    clusterer = _make_mock_clusterer(0)  # Budget
    result = display_tier(clusterer, MOCK_LABEL_MAP, MOCK_CLUSTER_STATS, bedrooms=2, area_m2=80, low=90_000, high=140_000)
    assert 'Budget' in result
    # stats for Budget tier should appear in the output
    assert '80k' in result or '80,000' in result or '80K' in result or '$80' in result


def test_display_tier_luxury_for_high_price():
    clusterer = _make_mock_clusterer(2)  # Luxury
    result = display_tier(clusterer, MOCK_LABEL_MAP, MOCK_CLUSTER_STATS, bedrooms=4, area_m2=200, low=350_000, high=450_000)
    assert 'Luxury' in result


def test_display_tier_budget_for_low_price():
    clusterer = _make_mock_clusterer(0)  # Budget
    result = display_tier(clusterer, MOCK_LABEL_MAP, MOCK_CLUSTER_STATS, bedrooms=1, area_m2=55, low=85_000, high=120_000)
    assert 'Budget' in result


def test_display_tier_midpoint_passed_to_clusterer():
    clusterer = _make_mock_clusterer(1)
    display_tier(clusterer, MOCK_LABEL_MAP, MOCK_CLUSTER_STATS, bedrooms=3, area_m2=120, low=160_000, high=200_000)
    call_arg = clusterer.predict.call_args[0][0]
    # midpoint of 160k–200k is 180k; input should be a DataFrame with bedrooms/area_m2/price columns
    assert list(call_arg.columns) == ['bedrooms', 'area_m2', 'price']
    assert call_arg.iloc[0]['bedrooms'] == 3
    assert call_arg.iloc[0]['area_m2'] == 120
    assert call_arg.iloc[0]['price'] == 180_000.0


# --- predict_tier tests ---

def test_predict_tier_returns_dict_with_required_keys():
    clusterer = _make_mock_clusterer(1)  # Mid-Range
    result = predict_tier(clusterer, MOCK_LABEL_MAP, MOCK_CLUSTER_STATS, bedrooms=3, area_m2=120, low=160_000, high=200_000)
    assert isinstance(result, dict)
    assert 'market_tier' in result
    assert 'tier_stats' in result


def test_predict_tier_market_tier_is_valid_label():
    clusterer = _make_mock_clusterer(0)  # Budget
    result = predict_tier(clusterer, MOCK_LABEL_MAP, MOCK_CLUSTER_STATS, bedrooms=2, area_m2=80, low=90_000, high=140_000)
    assert result['market_tier'] in ('Budget', 'Mid-Range', 'Luxury')


def test_predict_tier_stats_has_verbatim_artifact_keys():
    clusterer = _make_mock_clusterer(2)  # Luxury
    result = predict_tier(clusterer, MOCK_LABEL_MAP, MOCK_CLUSTER_STATS, bedrooms=4, area_m2=200, low=350_000, high=450_000)
    stats = result['tier_stats']
    for key in ('price_p10', 'price_p90', 'area_p10', 'area_p90', 'beds_p10', 'beds_p90'):
        assert key in stats, f"Missing key: {key}"


def test_predict_tier_midpoint_passed_correctly():
    clusterer = _make_mock_clusterer(1)
    predict_tier(clusterer, MOCK_LABEL_MAP, MOCK_CLUSTER_STATS, bedrooms=3, area_m2=120, low=160_000, high=200_000)
    call_arg = clusterer.predict.call_args[0][0]
    assert list(call_arg.columns) == ['bedrooms', 'area_m2', 'price']
    assert call_arg.iloc[0]['price'] == 180_000.0
