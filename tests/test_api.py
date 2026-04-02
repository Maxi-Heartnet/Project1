import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


MOCK_ARTIFACT = {
    'model': MagicMock(),
    'encoder': MagicMock(),
    'clusterer': MagicMock(),
    'cluster_label_map': {0: 'Budget', 1: 'Mid-Range', 2: 'Luxury'},
    'cluster_stats': {
        'Budget':    {'price_p10': 80_000,  'price_p90': 150_000, 'area_p10': 50,  'area_p90': 100, 'beds_p10': 1, 'beds_p90': 2},
        'Mid-Range': {'price_p10': 150_000, 'price_p90': 280_000, 'area_p10': 90,  'area_p90': 160, 'beds_p10': 2, 'beds_p90': 3},
        'Luxury':    {'price_p10': 280_000, 'price_p90': 500_000, 'area_p10': 140, 'area_p90': 300, 'beds_p10': 3, 'beds_p90': 5},
    },
}


def _setup_mocks():
    """Configure mock model to return predictable values."""
    mock_tree = MagicMock()
    mock_tree.predict.return_value = np.array([150_000.0])
    MOCK_ARTIFACT['model'].estimators_ = [mock_tree] * 20
    MOCK_ARTIFACT['encoder'].transform.return_value = np.zeros((1, 8))
    MOCK_ARTIFACT['clusterer'].predict.return_value = np.array([1])  # Mid-Range


@pytest.fixture
def client():
    _setup_mocks()
    with patch('api.joblib') as mock_joblib:
        mock_joblib.load.return_value = MOCK_ARTIFACT
        from api import app
        with TestClient(app) as c:
            yield c


# --- Health endpoint ---

def test_health_returns_200(client):
    response = client.get('/health')
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'ok'
    assert data['model_loaded'] is True


# --- Happy path ---

def test_predict_returns_200_with_valid_input(client):
    response = client.post('/predict', json={
        'sector': 'Piantini', 'property_type': 'apartment',
        'bedrooms': 3, 'area_m2': 120.0,
    })
    assert response.status_code == 200


def test_predict_response_has_required_keys(client):
    response = client.post('/predict', json={
        'sector': 'Piantini', 'property_type': 'apartment',
        'bedrooms': 3, 'area_m2': 120.0,
    })
    data = response.json()
    assert 'price_low' in data
    assert 'price_high' in data
    assert 'market_tier' in data
    assert 'tier_stats' in data


def test_predict_price_low_lte_high(client):
    response = client.post('/predict', json={
        'sector': 'Piantini', 'property_type': 'apartment',
        'bedrooms': 3, 'area_m2': 120.0,
    })
    data = response.json()
    assert data['price_low'] <= data['price_high']


def test_predict_market_tier_is_valid(client):
    response = client.post('/predict', json={
        'sector': 'Piantini', 'property_type': 'apartment',
        'bedrooms': 3, 'area_m2': 120.0,
    })
    assert response.json()['market_tier'] in ('Budget', 'Mid-Range', 'Luxury')


def test_predict_tier_stats_has_verbatim_keys(client):
    response = client.post('/predict', json={
        'sector': 'Piantini', 'property_type': 'apartment',
        'bedrooms': 3, 'area_m2': 120.0,
    })
    stats = response.json()['tier_stats']
    for key in ('price_p10', 'price_p90', 'area_p10', 'area_p90', 'beds_p10', 'beds_p90'):
        assert key in stats


# --- Normalization (R11) ---

def test_predict_sector_normalized_to_title_case(client):
    response = client.post('/predict', json={
        'sector': 'piantini', 'property_type': 'apartment',
        'bedrooms': 3, 'area_m2': 120.0,
    })
    assert response.status_code == 200


def test_predict_property_type_normalized_to_lowercase(client):
    response = client.post('/predict', json={
        'sector': 'Piantini', 'property_type': 'Apartment',
        'bedrooms': 3, 'area_m2': 120.0,
    })
    assert response.status_code == 200


# --- Validation errors (R4) ---

def test_predict_missing_area_m2_returns_422(client):
    response = client.post('/predict', json={
        'sector': 'Piantini', 'property_type': 'apartment', 'bedrooms': 3,
    })
    assert response.status_code == 422


def test_predict_bedrooms_zero_returns_422(client):
    response = client.post('/predict', json={
        'sector': 'Piantini', 'property_type': 'apartment',
        'bedrooms': 0, 'area_m2': 120.0,
    })
    assert response.status_code == 422


def test_predict_negative_area_returns_422(client):
    response = client.post('/predict', json={
        'sector': 'Piantini', 'property_type': 'apartment',
        'bedrooms': 3, 'area_m2': -10.0,
    })
    assert response.status_code == 422


def test_predict_unknown_property_type_returns_200(client):
    # handle_unknown='ignore' in encoder — not a validation error
    response = client.post('/predict', json={
        'sector': 'Piantini', 'property_type': 'villa',
        'bedrooms': 3, 'area_m2': 120.0,
    })
    assert response.status_code == 200
