from pydantic import BaseModel


class CarFeatures(BaseModel):
    make: str
    model: str
    year: int
    style: str
    distance: float
    engine_capacity: float
    fuel_type: str
    transmission: str


class PredictionResponse(BaseModel):
    prediction_id: int
    predicted_price: float