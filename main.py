import asyncio, json, os
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from agents.zone_agent import generate_safe_zones
from agents.civilian_agent import get_batch_decisions, get_all_decisions_batch
from agents.coordinator_agent import get_coordinator_assessment
from models.hurricane import HurricaneInput
from models.agent import SimulationStartInput, AgentDecisionsInput
from simulation.engine import engine
from simulation.routing import assign_agents_to_zones

load_dotenv()
# google-adk reads GOOGLE_API_KEY; support legacy GEMINI_API_KEY as fallback
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY", "")

class ConnectionManager:
    def __init__(self):
        self.active = []
    async def connect(self, ws):
        await ws.accept()
        self.active.append(ws)
    def disconnect(self, ws):
        if ws in self.active: self.active.remove(ws)
    async def broadcast(self, data):
        message = json.dumps(data)
        dead = []
        for ws in self.active:
            try: await ws.send_text(message)
            except: dead.append(ws)
        for ws in dead: self.disconnect(ws)

manager = ConnectionManager()

app = FastAPI(title="Hurricane Evac API")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://localhost:5174", "hurricane-simulator-fron-git-f68011-varunmuchanapallys-projects.vercel.app", "hurricane-simulator-frontend-n7ch-3gkeg3gx2.vercel.app"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/api/generate-safe-zones")
async def api_generate_safe_zones(payload: HurricaneInput):
    zones = await generate_safe_zones(payload.origin_lng, payload.origin_lat, payload.dest_lng, payload.dest_lat, payload.category, payload.wind_speed)
    return {"safe_zones": zones}

@app.post("/api/assign-routes")
async def api_assign_routes(payload: dict):
    """Route Agent assigns civilians to zones; Mapbox provides road geometry."""
    results = await assign_agents_to_zones(
        payload.get("agents", []),
        payload.get("safe_zones", []),
        payload.get("hurricane"),
    )
    return {"assignments": results}

@app.post("/api/agent-decisions")
async def api_agent_decisions(payload: AgentDecisionsInput):
    decisions = await get_all_decisions_batch(payload.agents, payload.hurricane, payload.safe_zones)
    return {"decisions": decisions}

@app.post("/api/start-simulation")
async def api_start_simulation(payload: SimulationStartInput):
    if engine.running:
        engine.stop()
        await asyncio.sleep(0.2)

    agents_raw = [a.model_dump() for a in payload.agents]
    safe_zones = payload.safe_zones
    hurricane  = payload.hurricane

    assignments = await assign_agents_to_zones(agents_raw, safe_zones)
    assignment_map = {a["agent_id"]: a for a in assignments}

    agents_assigned = []
    for agent in agents_raw:
        asgn = assignment_map.get(agent["id"], {})
        agents_assigned.append({**agent, "assignedZoneId": asgn.get("zone_id"), "route": asgn.get("route"), "distanceKm": asgn.get("distance_km"), "etaHours": asgn.get("eta_hours"), "status": asgn.get("status", "waiting"), "progress": 0.0})

    decisions = await get_batch_decisions(agents_assigned, hurricane, safe_zones)
    decision_map = {d["agent_id"]: d for d in decisions}
    for agent in agents_assigned:
        action = decision_map.get(agent["id"], {}).get("action", "evacuate")
        if action == "shelter_in_place": agent["status"] = "waiting"; agent["route"] = None
        elif action == "request_help":   agent["status"] = "stranded"

    engine.setup(hurricane, agents_assigned, safe_zones)
    engine.set_broadcast(manager.broadcast)
    asyncio.create_task(engine.run())

    await manager.broadcast({"type": "simulation_started", "agent_count": len(agents_assigned), "zone_count": len(safe_zones)})
    return {"status": "started", "agents": agents_assigned, "assignments": assignments, "decisions": decisions}

@app.post("/api/pause")
async def api_pause():
    engine.pause()
    await manager.broadcast({"type": "status", "status": "paused"})
    return {"status": "paused"}

@app.post("/api/resume")
async def api_resume():
    engine.resume()
    await manager.broadcast({"type": "status", "status": "running"})
    return {"status": "running"}

@app.post("/api/reset")
async def api_reset():
    engine.stop()
    await manager.broadcast({"type": "status", "status": "idle"})
    return {"status": "reset"}

@app.get("/api/state")
async def api_state():
    return engine.get_state()

@app.post("/api/coordinator-check")
async def api_coordinator_check():
    assessment = await get_coordinator_assessment(engine.agents, engine.safe_zones, engine.hurricane or {}, engine.elapsed_hours)
    await manager.broadcast({"type": "coordinator_assessment", **assessment})
    for alert in assessment.get("alerts", []):
        await manager.broadcast({"type": "log", "message": f"COORDINATOR: {alert}", "level": "warning"})
    return assessment

@app.websocket("/ws/simulation")
async def ws_simulation(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await websocket.send_text(json.dumps({"type": "connected", "state": engine.get_state()}))
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            elif msg.get("type") == "set_speed":
                engine.speed = float(msg.get("speed", 1.0))
    except WebSocketDisconnect:
        manager.disconnect(websocket)