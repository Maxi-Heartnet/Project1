import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from chatbot.chat import ask_string, ask_number, collect_features, predict_range


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
