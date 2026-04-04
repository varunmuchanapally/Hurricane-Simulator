import asyncio
from typing import List, Optional, Callable
from simulation.hurricane import build_control_points, interpolate_position, haversine_km

class SimulationEngine:
    def __init__(self):
        self.running = False
        self.paused = False
        self.tick = 0
        self.elapsed_hours = 0.0
        self.speed = 1.0
        self.progress = 0.0
        self.hurricane = None
        self.control_points = None
        self.agents: List[dict] = []
        self.safe_zones: List[dict] = []
        self.logs: List[dict] = []
        self._broadcast_cb: Optional[Callable] = None

    def setup(self, hurricane, agents, safe_zones):
        self.hurricane = hurricane
        self.control_points = build_control_points(
            hurricane["originLng"], hurricane["originLat"],
            hurricane["destLng"],   hurricane["destLat"],
        )
        self.agents = agents
        self.safe_zones = safe_zones
        self.tick = 0
        self.elapsed_hours = 0.0
        self.progress = 0.0

    def set_broadcast(self, cb):
        self._broadcast_cb = cb

    async def run(self):
        self.running = True
        while self.running and self.progress < 1.0:
            if not self.paused:
                await self._tick()
            await asyncio.sleep(0.1)
        self.running = False
        await self._broadcast({"type": "status", "status": "complete"})

    async def _tick(self):
        self.tick += 1
        self.elapsed_hours = round(self.elapsed_hours + 0.05 * self.speed, 2)
        self.progress = min(self.progress + 0.001 * self.speed, 1.0)
        pos = interpolate_position(self.control_points, self.progress)
        hurricane_pos = {"lng": pos["lng"], "lat": pos["lat"]}

        updated_agents = []
        arrivals = []
        for agent in self.agents:
            updated = self._move_agent(agent, hurricane_pos)
            updated_agents.append(updated)
            if updated["status"] == "arrived" and agent["status"] == "evacuating":
                arrivals.append(updated)
        self.agents = updated_agents

        await self._broadcast({
            "type": "tick",
            "tick": self.tick,
            "elapsed_hours": self.elapsed_hours,
            "hurricane_progress": round(self.progress, 4),
            "hurricane_position": hurricane_pos,
            "agents": [{"id": a["id"], "position": a["position"], "status": a["status"], "progress": a.get("progress", 0)} for a in self.agents],
            "arrivals": [a["id"] for a in arrivals],
        })

        for a in arrivals:
            await self._broadcast({"type": "log", "message": f"{a['label']} agent arrived safely", "level": "success", "elapsed": self.elapsed_hours})

        for agent in self.agents:
            if agent["status"] == "evacuating":
                dist = haversine_km(agent["position"]["lat"], agent["position"]["lng"], hurricane_pos["lat"], hurricane_pos["lng"])
                if dist < 50:
                    await self._broadcast({"type": "log", "message": f"WARNING: {agent['label']} in {agent['neighborhood']} is {round(dist)}km from eye", "level": "warning", "elapsed": self.elapsed_hours})

    def _move_agent(self, agent, hurricane_pos):
        if agent["status"] in ("arrived", "stranded"):
            return agent
        route = agent.get("route")
        if not route or len(route) < 2:
            return agent
        step = 0.008 * agent.get("speed", 0.7) * self.speed
        new_progress = min(agent.get("progress", 0) + step, 1.0)
        route_len = len(route) - 1
        route_t = new_progress * route_len
        route_idx = min(int(route_t), route_len - 1)
        route_frac = route_t - route_idx
        frm = route[route_idx]
        to  = route[min(route_idx + 1, route_len)]
        lng = frm[0] + (to[0] - frm[0]) * route_frac
        lat = frm[1] + (to[1] - frm[1]) * route_frac
        new_status = "arrived" if new_progress >= 1.0 else "evacuating"
        return {**agent, "progress": round(new_progress, 4), "position": {"lng": round(lng, 6), "lat": round(lat, 6)}, "status": new_status}

    async def _broadcast(self, payload):
        if self._broadcast_cb:
            await self._broadcast_cb(payload)

    def pause(self):  self.paused = True
    def resume(self): self.paused = False
    def stop(self):   self.running = False

    def get_state(self):
        return {"running": self.running, "tick": self.tick, "elapsed_hours": self.elapsed_hours, "progress": self.progress, "agents": self.agents, "safe_zones": self.safe_zones, "logs": self.logs[-50:]}

engine = SimulationEngine()