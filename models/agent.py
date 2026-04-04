from pydantic import BaseModel
from typing import Optional, List

class Position(BaseModel):
    lng: float
    lat: float

class AgentInput(BaseModel):
    id: str
    type: str
    label: str
    neighborhood: str
    position: Position
    # Rich character fields
    name: Optional[str] = None
    age: Optional[int] = None
    situation: Optional[str] = None
    personality: Optional[str] = None
    # Assignment & status
    assigned_zone_id: Optional[str] = None
    assignedZoneId: Optional[str] = None
    needs_medical: bool = False
    needsMedical: bool = False
    speed: float = 0.7
    status: Optional[str] = "waiting"
    progress: Optional[float] = 0.0
    route: Optional[list] = None
    distance_km: Optional[float] = None
    eta_hours: Optional[float] = None
    ai_decision: Optional[dict] = None

class AgentDecision(BaseModel):
    agent_id: str
    action: str
    reasoning: str
    urgency: str
    message: Optional[str] = None

class AgentDecisionsInput(BaseModel):
    hurricane: dict
    agents: List[dict]
    safe_zones: List[dict]

class SimulationStartInput(BaseModel):
    hurricane: dict
    agents: List[AgentInput]
    safe_zones: List[dict]
