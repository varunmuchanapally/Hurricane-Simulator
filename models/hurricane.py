from pydantic import BaseModel
from typing import Optional

class HurricaneInput(BaseModel):
    origin_lng: float
    origin_lat: float
    dest_lng: float
    dest_lat: float
    category: int
    wind_speed: float

class HurricanePosition(BaseModel):
    lng: float
    lat: float
    progress: float