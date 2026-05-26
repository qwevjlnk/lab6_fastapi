import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from sklearn.preprocessing import OrdinalEncoder
from sqlalchemy.orm import Session

from app.database import Base, engine, get_db
from app.models import Prediction
from app.schemas import CarFeatures, PredictionResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

CAR_MODEL_PATH = BASE_DIR / "cars.joblib"
PRICE_ENCODER_PATH = BASE_DIR / "power.joblib"


class DummyModel:
    def predict(self, x):
        return np.array([0])


class DummyPriceEncoder:
    def inverse_transform(self, x):
        return np.asarray(x, dtype=float)


def load_artifact(path: Path, fallback_factory):
    try:
        artifact = joblib.load(path)
        logger.info("Loaded artifact: %s", path.name)
        return artifact
    except Exception as e:
        logger.warning("Failed to load %s: %s", path.name, e)
        return fallback_factory()


ml_model = load_artifact(CAR_MODEL_PATH, DummyModel)
price_decoder = load_artifact(PRICE_ENCODER_PATH, DummyPriceEncoder)

app = FastAPI(
    title="Car Price API",
    version="1.0.0"
)

# Создание таблиц БД
Base.metadata.create_all(bind=engine)


def clear_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Кодирование категориальных признаков
    """
    df = df.copy()

    cat_columns = [
        "Make",
        "Model",
        "Style",
        "Fuel_type",
        "Transmission"
    ]

    ordinal = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    df[cat_columns] = ordinal.fit_transform(df[cat_columns])

    return df


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Генерация новых признаков
    """
    df = df.copy()

    years_left = (2025 - df["Year"]).replace(0, 1)

    df["Distance_by_year"] = df["Distance"] / years_left
    df["age"] = 2025 - df["Year"]

    mean_engine_cap = (
        df.groupby("Style")["Engine_capacity"]
        .transform("mean")
    )

    max_engine_cap = (
        df.groupby("Style")["Engine_capacity"]
        .transform("max")
    )

    df["eng_cap_diff"] = (
        df["Engine_capacity"] - mean_engine_cap
    ).abs()

    df["eng_cap_diff_max"] = (
        df["Engine_capacity"] - max_engine_cap
    ).abs()

    return df


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "car-price-api"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict car price"
)
def predict(
    car: CarFeatures,
    db: Session = Depends(get_db)
):
    try:

        rename_map = {
            "make": "Make",
            "model": "Model",
            "year": "Year",
            "style": "Style",
            "distance": "Distance",
            "engine_capacity": "Engine_capacity",
            "fuel_type": "Fuel_type",
            "transmission": "Transmission",
        }

        input_data = pd.DataFrame([car.model_dump()])
        input_data = input_data.rename(columns=rename_map)

        prepared = featurize(clear_data(input_data))

        raw_pred = ml_model.predict(prepared)

        raw_pred = np.asarray(raw_pred).reshape(-1, 1)

        price_array = price_decoder.inverse_transform(raw_pred)

        predicted_price = float(
            np.asarray(price_array).ravel()[0]
        )

        # Сохраняем prediction в БД
        row = Prediction(
            make=car.make,
            model=car.model,
            year=car.year,
            style=car.style,
            distance=car.distance,
            engine_capacity=car.engine_capacity,
            fuel_type=car.fuel_type,
            transmission=car.transmission,
            predicted_price=predicted_price,
        )

        db.add(row)
        db.commit()
        db.refresh(row)

        return PredictionResponse(
            prediction_id=row.id,
            predicted_price=round(predicted_price, 2),
        )

    except Exception as e:
        logger.exception("Prediction error")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/predictions")
def list_predictions(
    db: Session = Depends(get_db)
):
    rows = (
        db.query(Prediction)
        .order_by(Prediction.id.desc())
        .all()
    )

    return [
        {
            "prediction_id": row.id,
            "make": row.make,
            "model": row.model,
            "year": row.year,
            "style": row.style,
            "distance": row.distance,
            "engine_capacity": row.engine_capacity,
            "fuel_type": row.fuel_type,
            "transmission": row.transmission,
            "predicted_price": row.predicted_price,
            "created_at": row.created_at.isoformat(),
        }
        for row in rows
    ]