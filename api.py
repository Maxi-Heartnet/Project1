from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from chatbot.chat import predict_range, predict_tier
from ml.prepare import ALL_FEATURES

MODEL_PATH = Path(__file__).parent / 'ml' / 'model.pkl'

model_store = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifact = joblib.load(MODEL_PATH)
    model_store['model'] = artifact['model']
    model_store['encoder'] = artifact['encoder']
    model_store['clusterer'] = artifact['clusterer']
    model_store['cluster_label_map'] = artifact['cluster_label_map']
    model_store['cluster_stats'] = artifact['cluster_stats']
    yield
    model_store.clear()


app = FastAPI(
    title="Santo Domingo House Price API",
    description="Predicts house prices and market tier for properties in Santo Domingo, DR.",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    sector: str
    property_type: str
    bedrooms: int = Field(ge=1)
    area_m2: float = Field(gt=0)


class TierStats(BaseModel):
    price_p10: float
    price_p90: float
    area_p10: float
    area_p90: float
    beds_p10: float
    beds_p90: float


class PredictResponse(BaseModel):
    price_low: float
    price_high: float
    market_tier: str
    tier_stats: TierStats


@app.post('/predict', response_model=PredictResponse)
def predict(body: PredictRequest):
    sector = body.sector.title()
    property_type = body.property_type.lower()

    features = {
        'bedrooms': body.bedrooms,
        'area_m2': body.area_m2,
        'sector': sector,
        'property_type': property_type,
    }
    features_df = pd.DataFrame([features], columns=ALL_FEATURES)

    low, high = predict_range(model_store['model'], model_store['encoder'], features_df)

    tier_data = predict_tier(
        model_store['clusterer'],
        model_store['cluster_label_map'],
        model_store['cluster_stats'],
        bedrooms=body.bedrooms,
        area_m2=body.area_m2,
        low=float(low),
        high=float(high),
    )

    return PredictResponse(
        price_low=float(low),
        price_high=float(high),
        market_tier=tier_data['market_tier'],
        tier_stats=TierStats(**tier_data['tier_stats']),
    )


@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': 'model' in model_store}
