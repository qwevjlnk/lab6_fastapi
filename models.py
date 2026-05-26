from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String

from app.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)

    make = Column(String(100), nullable=False)
    model = Column(String(100), nullable=False)
    year = Column(Integer, nullable=False)
    style = Column(String(100), nullable=False)
    distance = Column(Float, nullable=False)
    engine_capacity = Column(Float, nullable=False)
    fuel_type = Column(String(50), nullable=False)
    transmission = Column(String(50), nullable=False)

    predicted_price = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)