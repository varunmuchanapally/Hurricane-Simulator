from pydantic import BaseModel
from typing import Optional

class Supplies(BaseModel):
    food: int
    water: int
    medical: int

class SafeZone(BaseModel):
    id: str
    name: str
    lat: float
    lng: float
    capacity: int
    supplies: Supplies
    reasoning: Optional[str] = None

class SafeZoneResponse(BaseModel):
    safe_zones: list[SafeZone]