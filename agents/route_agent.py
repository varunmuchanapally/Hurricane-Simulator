"""
Civilian AI Agents — each civilian is an autonomous ADK agent with two tools:

  get_hurricane_status()       — fetches live hurricane data (external source)
  get_routes_to_safe_zone()    — calls Mapbox Directions API (external source)

The civilian agent calls both tools, reads the real data, and decides:
  - Will I actually evacuate?
  - Which route would I realistically take given who I am?

80 civilians run as 80 parallel agent sessions.
"""

import json, re, uuid, os, math
import httpx
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# ── Shared session service ────────────────────────────────────────────────────
_session_service = InMemorySessionService()

# ── Module-level state set before agents run ─────────────────────────────────
_hurricane_state: dict = {}
_route_cache: dict     = {}   # agent_id → list of fetched route objects


# ── Saffir-Simpson physics (same values as hurricane_agent.py tools) ──────────
_WIND_RADIUS  = {1: 111, 2: 148, 3: 185, 4: 222, 5: 278}   # km, TS-force
_HF_RADIUS    = {1:  50, 2:  67, 3:  83, 4: 100, 5: 125}   # km, hurricane-force
_SURGE_TAMPA  = {1:   6, 2:  10, 3:  14, 4:  22, 5:  30}   # ft, bay-amplified


def set_hurricane_state(hurricane: dict):
    """Store live hurricane data, enriched with deterministic physics values."""
    global _hurricane_state
    cat  = max(1, min(5, int(hurricane.get("category", 3))))
    dist = hurricane.get("distanceKm", 0)

    # Threat level mirrors hurricane_agent.py logic
    if   cat >= 4 and dist < 200: threat = "extreme"
    elif cat >= 3 or dist < 150:  threat = "severe"
    elif cat >= 2 or dist < 300:  threat = "high"
    else:                          threat = "moderate"

    _hurricane_state = {
        **hurricane,
        "wind_radius_km":           _WIND_RADIUS[cat],
        "hurricane_force_radius_km": _HF_RADIUS[cat],
        "storm_surge_ft":           _SURGE_TAMPA[cat],
        "threat_level":             threat,
        "hours_to_landfall":        round(dist / 25, 1) if dist else None,
    }


# ── Haversine ─────────────────────────────────────────────────────────────────
def _haversine_km(lat1, lng1, lat2, lng2) -> float:
    R  = 6371
    dl = math.radians(lat2 - lat1)
    dg = math.radians(lng2 - lng1)
    a  = math.sin(dl/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dg/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Tool factories ────────────────────────────────────────────────────────────
# Each civilian agent gets its own bound tool instances so the tools know
# exactly which civilian's data to fetch without the LLM needing to pass coordinates.

def make_hurricane_tool():
    """Returns a tool that fetches the current hurricane status."""

    async def get_hurricane_status() -> str:
        """
        Fetch the current hurricane threat assessment from the emergency management system.
        Call this tool FIRST to understand the threat before deciding whether to evacuate.
        Returns: category, wind speed, distance, wind radius, storm surge, threat level, and guidance.
        """
        h = _hurricane_state
        if not h:
            return "Hurricane data unavailable. Treat situation as critical — evacuate immediately."

        dist_km    = h.get("distanceKm", 0)
        dist_miles = round(dist_km * 0.621371) if dist_km else "?"
        cat        = h.get("category", "?")
        wind       = h.get("windSpeed", "?")
        radius     = h.get("wind_radius_km", "?")
        surge      = h.get("storm_surge_ft", "?")
        threat     = h.get("threat_level", "high")
        eta        = h.get("hours_to_landfall")
        eta_str    = f"~{eta} hours to landfall" if eta else "landfall timing uncertain"

        return (
            f"THREAT LEVEL: {threat.upper()} | "
            f"Category {cat} hurricane, sustained winds {wind} mph. "
            f"Eye is {dist_miles} miles from Tampa Bay ({eta_str}). "
            f"Damaging winds extend {radius} km from the eye. "
            f"Storm surge forecast: up to {surge} ft — low-lying and coastal areas will flood. "
            f"Mandatory evacuation order in effect for all zones."
        )

    return get_hurricane_status


def make_route_tool(agent_id: str, origin_lng: float, origin_lat: float,
                    dest_lng: float, dest_lat: float, zone_name: str):
    """Returns a Mapbox-backed route tool bound to this civilian's origin and destination."""

    async def get_routes_to_safe_zone() -> str:
        f"""
        Fetch available driving routes to {zone_name} via the Mapbox Directions API.
        Call this tool to see what real roads exist before choosing your evacuation route.
        Returns road names, distances in km, and estimated driving times.
        """
        token = os.getenv("MAPBOX_TOKEN")

        # No token — describe a single direct route
        if not token:
            dist = _haversine_km(origin_lat, origin_lng, dest_lat, dest_lng)
            _route_cache[agent_id] = [{
                "index": 0, "summary": "Direct route",
                "distance_km": round(dist, 1), "duration_min": None, "coords": None,
            }]
            return f"Route 0 (direct road): approximately {round(dist, 1)} km to {zone_name}."

        url = (
            f"https://api.mapbox.com/directions/v5/mapbox/driving/"
            f"{origin_lng},{origin_lat};{dest_lng},{dest_lat}"
            f"?geometries=geojson&overview=full&alternatives=true&steps=true"
            f"&access_token={token}"
        )
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                res  = await client.get(url)
                data = res.json()

            raw_routes = data.get("routes", [])
            if not raw_routes:
                raise ValueError("No routes from Mapbox")

            fetched = []
            lines   = []
            for i, r in enumerate(raw_routes[:3]):
                legs    = r.get("legs", [{}])
                summary = legs[0].get("summary", "") if legs else ""
                if not summary:
                    steps   = legs[0].get("steps", []) if legs else []
                    names   = [s.get("name","") for s in steps[:3] if s.get("name")]
                    summary = ", ".join(names) or "local roads"
                dist_km  = round(r["distance"] / 1000, 1)
                dur_min  = round(r["duration"] / 60)
                is_hwy   = any(kw in summary.lower() for kw in
                               ["i-", "275", "us-", "fl-", "sr-", "interstate", "highway", "hwy"])
                road_tag = "highway" if is_hwy else "local roads"
                fetched.append({
                    "index": i, "summary": summary, "road_type": road_tag,
                    "distance_km": dist_km, "duration_min": dur_min,
                    "coords": r["geometry"]["coordinates"],
                })
                lines.append(
                    f"Route {i} — via {summary} ({road_tag}): {dist_km} km, ~{dur_min} min"
                )

            _route_cache[agent_id] = fetched
            return "\n".join(lines)

        except Exception:
            dist = _haversine_km(origin_lat, origin_lng, dest_lat, dest_lng)
            _route_cache[agent_id] = [{
                "index": 0, "summary": "Direct route",
                "distance_km": round(dist, 1), "duration_min": None, "coords": None,
            }]
            return f"Route 0 (direct): approximately {round(dist, 1)} km to {zone_name}."

    return get_routes_to_safe_zone


# ── Single civilian agent runner ──────────────────────────────────────────────

async def run_civilian_agent(civilian: dict, zone: dict) -> dict:
    """
    Run ONE civilian as an autonomous ADK agent.
    The agent calls tools to get hurricane data and Mapbox routes,
    then reasons and returns its decision.
    """
    pos       = civilian["position"]
    name      = civilian.get("name", "Unknown")
    age       = civilian.get("age", "?")
    ctype     = civilian.get("type", "adult")
    hood      = civilian.get("neighborhood", "Tampa")
    situation = civilian.get("situation", "N/A")
    persona   = civilian.get("personality", "rational")
    medical   = "yes" if (civilian.get("needsMedical") or civilian.get("needs_medical")) else "no"

    # Each civilian gets tools bound to their specific journey
    hurricane_tool = make_hurricane_tool()
    route_tool     = make_route_tool(
        civilian["id"],
        pos["lng"], pos["lat"],
        zone["lng"], zone["lat"],
        zone["name"],
    )

    agent = LlmAgent(
        name=f"civilian_{civilian['id'].replace('-','_')}",
        model="gemini-1.5-flash",
        tools=[hurricane_tool, route_tool],
        instruction=f"""You are {name}, a {age}-year-old {ctype} living in {hood}, Tampa Bay, Florida.

YOUR SITUATION: {situation}
YOUR PERSONALITY: {persona}
MEDICAL NEEDS: {medical}
ASSIGNED SAFE ZONE: {zone['name']}

A mandatory hurricane evacuation order has been issued for your area.
You are a real person, not an assistant. Think and respond as yourself.

YOUR DECISION PROCESS:
  1. Call get_hurricane_status() to learn the real threat — category, wind radius, storm surge,
     threat level, and how many hours until landfall. This is live data from the emergency system.
  2. Call get_routes_to_safe_zone() to see actual roads available to you.
  3. Based on who you are, your situation, the real threat level, and your available routes — decide.

REALISTIC BEHAVIOR GUIDE:
  - threat_level EXTREME or SEVERE: most people leave, even stubborn ones feel the pressure
  - threat_level HIGH: mixed — some leave, some wait and see
  - threat_level MODERATE: many delay or shelter in place
  - People with medical needs or no transport are more likely to request_help
  - Stubborn/overconfident personalities may shelter_in_place even in severe threats
  - Panicked personalities may evacuate but choose a poor route (higher index = less optimal)

Not everyone leaves. Some people refuse. Some can't. Be authentic.

Return ONLY valid JSON — no markdown, no explanation:
{{"action": "evacuate|shelter_in_place|request_help",
  "route_index": 0,
  "reasoning": "one sentence in your own voice explaining your choice",
  "message": "short message you'd broadcast on emergency radio, or null"}}""",
    )

    runner    = Runner(agent=agent, app_name="hurricane_evac", session_service=_session_service)
    sess_id   = str(uuid.uuid4())
    await _session_service.create_session(
        app_name="hurricane_evac", user_id=civilian["id"], session_id=sess_id
    )

    trigger_msg = types.Content(
        role="user",
        parts=[types.Part(text="The evacuation order has been issued. What do you do?")]
    )

    response_text = ""
    async for event in runner.run_async(
        user_id=civilian["id"], session_id=sess_id, new_message=trigger_msg
    ):
        if event.is_final_response() and event.content and event.content.parts:
            response_text = event.content.parts[0].text

    try:
        text     = re.sub(r"```json\s*|```\s*", "", response_text.strip()).strip()
        decision = json.loads(text)
        decision["agent_id"] = civilian["id"]
        return decision
    except Exception:
        return {
            "agent_id":   civilian["id"],
            "action":     "evacuate",
            "route_index": 0,
            "reasoning":  "Following the official evacuation order.",
            "message":    None,
        }


# ── Zone assignment agent (unchanged) ────────────────────────────────────────

_zone_agent = LlmAgent(
    name="zone_assignment_agent",
    model="gemini-1.5-flash",
    description="Assigns evacuating Tampa Bay civilians to the most appropriate safe zone",
    instruction="""You are the Tampa Bay Emergency Evacuation Zone Coordinator.
Assign each civilian to exactly one safe zone. Treat every person as a real individual.

Rules:
1. Civilians with medical needs MUST go to a zone with medical supplies > 0
2. Disabled and elderly go to the CLOSEST zone with medical supplies
3. Distribute load — no zone should exceed 85% capacity unless unavoidable
4. Families with children should avoid the most crowded zones
5. No car or limited mobility → assign nearest zone
6. Every civilian must receive an assignment

Return ONLY valid JSON array, no markdown:
[{"agent_id": "...", "zone_id": "...", "reasoning": "one sentence why this zone fits this person"}]""",
)

_zone_runner = Runner(
    agent=_zone_agent, app_name="hurricane_evac", session_service=_session_service
)


async def assign_zones_via_agent(agents: list, safe_zones: list) -> list:
    """Zone Agent assigns every civilian to the most appropriate safe zone."""
    zone_lines  = [
        f"  {z['id']}: {z['name']} | capacity {z['capacity']} | "
        f"medical: {z.get('supplies',{}).get('medical',0)} units"
        for z in safe_zones
    ]
    agent_lines = []
    for a in agents:
        pos    = a["position"]
        dists  = ", ".join(
            f"{z['name']} {round(_haversine_km(pos['lat'],pos['lng'],z['lat'],z['lng']))}km"
            for z in safe_zones
        )
        med = "yes" if (a.get("needsMedical") or a.get("needs_medical")) else "no"
        agent_lines.append(
            f"  {a['id']} | {a.get('name','Unknown')} | {a.get('type','adult')}, "
            f"age {a.get('age','?')} | {a.get('neighborhood','Tampa')} | "
            f"medical: {med} | Situation: {a.get('situation','N/A')} | "
            f"Personality: {a.get('personality','rational')} | "
            f"Distances → {dists}"
        )

    prompt = (
        f"Assign {len(agents)} civilians to {len(safe_zones)} safe zones.\n\n"
        f"SAFE ZONES:\n" + "\n".join(zone_lines) + "\n\n"
        f"CIVILIANS:\n" + "\n".join(agent_lines) + "\n\n"
        f"Return JSON array only."
    )

    sess_id = str(uuid.uuid4())
    await _session_service.create_session(
        app_name="hurricane_evac", user_id="system", session_id=sess_id
    )
    content = types.Content(role="user", parts=[types.Part(text=prompt)])

    response_text = ""
    async for event in _zone_runner.run_async(
        user_id="system", session_id=sess_id, new_message=content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            response_text = event.content.parts[0].text

    try:
        data        = json.loads(re.sub(r"```json\s*|```\s*", "", response_text.strip()).strip())
        assigned    = {a["agent_id"] for a in data}
        for agent in agents:
            if agent["id"] not in assigned:
                best = max(safe_zones, key=lambda z: z["capacity"])
                data.append({"agent_id": agent["id"], "zone_id": best["id"],
                              "reasoning": "Fallback to highest-capacity zone."})
        return data
    except Exception:
        return _fallback_nearest(agents, safe_zones)


def _fallback_nearest(agents, safe_zones):
    return [
        {
            "agent_id":  a["id"],
            "zone_id":   min(safe_zones, key=lambda z: _haversine_km(
                             a["position"]["lat"], a["position"]["lng"], z["lat"], z["lng"]))["id"],
            "reasoning": "Nearest available zone.",
        }
        for a in agents
    ]
