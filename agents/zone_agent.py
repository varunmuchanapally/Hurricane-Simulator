"""
Safe Zone Agent — Core Agent 2.

Agent2Agent pipeline:
  Safe Zone Agent  →  (AgentTool)  →  Hurricane Analysis Agent
                                       ├─ calls get_wind_radius()
                                       └─ calls get_storm_surge_ft()
                   ←  threat assessment (wind radius, surge, safe directions)
  Safe Zone Agent selects the best zones from a REAL candidate pool.

The AI never invents coordinates — it reasons over a hardcoded list of real
Florida venues and picks the optimal subset given the storm trajectory,
threat level, civilian needs, and capacity requirements.

Model: gemini-2.5-flash — has built-in chain-of-thought thinking, which
means it will reason step-by-step through the threat data before committing
to a zone selection.
"""

import json, re, uuid
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.agent_tool import AgentTool
from google.genai import types
from simulation.hurricane import build_control_points, get_path_sample_points, haversine_km
from agents.hurricane_agent import hurricane_analysis_agent


_session_service = InMemorySessionService()


# ── Candidate pool — real Florida venues with verified coordinates ─────────────
# direction_from_tampa: compass quadrant relative to Tampa Bay center (27.95, -82.46)
# is_inland: True = not in coastal flood zone, False = coastal/low-lying
# has_medical: whether the venue has or is near a medical facility

CANDIDATE_ZONES = [
    # ── Tampa metro (usable when storm approaches from far south/southeast) ──
    {
        "id": "cz01",
        "name": "Florida State Fairgrounds, Tampa",
        "lat": 27.9828, "lng": -82.3637,
        "max_capacity": 800,
        "direction": "east",
        "is_inland": True,
        "has_medical": False,
        "notes": "Large fairground complex east of downtown Tampa, above surge elevation",
    },
    {
        "id": "cz02",
        "name": "Hillsborough Community College – Brandon Campus",
        "lat": 27.9317, "lng": -82.2955,
        "max_capacity": 500,
        "direction": "east",
        "is_inland": True,
        "has_medical": False,
        "notes": "Inland campus in Brandon, 12 miles east of Tampa",
    },
    {
        "id": "cz03",
        "name": "Raymond James Stadium, Tampa",
        "lat": 27.9759, "lng": -82.5033,
        "max_capacity": 900,
        "direction": "northwest",
        "is_inland": True,
        "has_medical": False,
        "notes": "Large NFL stadium — only safe if storm approaches from the east; avoid for Gulf-approach storms",
    },
    # ── North corridor ─────────────────────────────────────────────────────────
    {
        "id": "cz04",
        "name": "Yuengling Center – University of South Florida, Tampa",
        "lat": 28.0655, "lng": -82.4149,
        "max_capacity": 600,
        "direction": "north",
        "is_inland": True,
        "has_medical": True,
        "notes": "University arena with on-campus medical resources; north of downtown",
    },
    {
        "id": "cz05",
        "name": "Pasco-Hernando State College – Wesley Chapel",
        "lat": 28.1935, "lng": -82.3593,
        "max_capacity": 450,
        "direction": "north",
        "is_inland": True,
        "has_medical": False,
        "notes": "Community college campus 20 miles north, inland Pasco County",
    },
    {
        "id": "cz06",
        "name": "Wiregrass Ranch Sports Campus, Wesley Chapel",
        "lat": 28.1780, "lng": -82.3460,
        "max_capacity": 550,
        "direction": "north",
        "is_inland": True,
        "has_medical": False,
        "notes": "Large multi-use sports complex in northern Hillsborough/Pasco",
    },
    {
        "id": "cz07",
        "name": "Hernando County Fairgrounds, Brooksville",
        "lat": 28.5575, "lng": -82.3832,
        "max_capacity": 650,
        "direction": "north",
        "is_inland": True,
        "has_medical": False,
        "notes": "Spacious fairground 40 miles north; elevated terrain well above surge",
    },
    {
        "id": "cz08",
        "name": "Ocala Civic Theater Complex, Ocala",
        "lat": 29.1872, "lng": -82.1401,
        "max_capacity": 700,
        "direction": "north",
        "is_inland": True,
        "has_medical": True,
        "notes": "60 miles north, near Munroe Regional Medical Center; high ground",
    },
    # ── Northeast corridor ─────────────────────────────────────────────────────
    {
        "id": "cz09",
        "name": "The Villages – Sumter Landing Community Center",
        "lat": 28.9025, "lng": -82.0197,
        "max_capacity": 700,
        "direction": "northeast",
        "is_inland": True,
        "has_medical": True,
        "notes": "Large retirement community with its own medical facilities; 55 miles NE",
    },
    {
        "id": "cz10",
        "name": "Marion County Fairgrounds, Ocala",
        "lat": 29.1300, "lng": -82.1370,
        "max_capacity": 800,
        "direction": "northeast",
        "is_inland": True,
        "has_medical": False,
        "notes": "Large county fairground on high inland ground, 65 miles NE",
    },
    {
        "id": "cz11",
        "name": "University of Florida – O'Connell Center, Gainesville",
        "lat": 29.6516, "lng": -82.3486,
        "max_capacity": 1000,
        "direction": "northeast",
        "is_inland": True,
        "has_medical": True,
        "notes": "90 miles NE; UF Shands Hospital adjacent — ideal for medical-needs civilians",
    },
    # ── East corridor ──────────────────────────────────────────────────────────
    {
        "id": "cz12",
        "name": "Strawberry Festival Grounds, Plant City",
        "lat": 28.0189, "lng": -82.1143,
        "max_capacity": 700,
        "direction": "east",
        "is_inland": True,
        "has_medical": False,
        "notes": "Large fairground 25 miles east; well inland from Tampa Bay",
    },
    {
        "id": "cz13",
        "name": "RP Funding Center (Civic Center), Lakeland",
        "lat": 28.0395, "lng": -81.9498,
        "max_capacity": 600,
        "direction": "east",
        "is_inland": True,
        "has_medical": True,
        "notes": "35 miles east; Lakeland Regional Health nearby",
    },
    {
        "id": "cz14",
        "name": "Polk County Fairgrounds, Bartow",
        "lat": 27.8978, "lng": -81.8430,
        "max_capacity": 750,
        "direction": "east",
        "is_inland": True,
        "has_medical": False,
        "notes": "45 miles east of Tampa, elevated central Florida terrain",
    },
    {
        "id": "cz15",
        "name": "Osceola Heritage Park, Kissimmee",
        "lat": 28.2960, "lng": -81.4073,
        "max_capacity": 800,
        "direction": "east",
        "is_inland": True,
        "has_medical": False,
        "notes": "65 miles east; large event complex on high ground near Orlando",
    },
    # ── Southeast/far east ─────────────────────────────────────────────────────
    {
        "id": "cz16",
        "name": "Orange County Convention Center, Orlando",
        "lat": 28.4253, "lng": -81.4711,
        "max_capacity": 1200,
        "direction": "east",
        "is_inland": True,
        "has_medical": False,
        "notes": "Massive convention center 75 miles east; highest capacity in pool",
    },
    {
        "id": "cz17",
        "name": "Amway Center, Orlando",
        "lat": 28.5392, "lng": -81.3839,
        "max_capacity": 900,
        "direction": "east",
        "is_inland": True,
        "has_medical": True,
        "notes": "Downtown Orlando arena; near Orlando Health hospital",
    },
    {
        "id": "cz18",
        "name": "Addition Financial Arena – UCF, Orlando",
        "lat": 28.6024, "lng": -81.1999,
        "max_capacity": 800,
        "direction": "east",
        "is_inland": True,
        "has_medical": True,
        "notes": "UCF campus east of Orlando; UCF Health on-site",
    },
    {
        "id": "cz19",
        "name": "Lake County Fairgrounds, Eustis",
        "lat": 28.8013, "lng": -81.6793,
        "max_capacity": 600,
        "direction": "northeast",
        "is_inland": True,
        "has_medical": False,
        "notes": "55 miles northeast; rural inland fairground above flood zones",
    },
    {
        "id": "cz20",
        "name": "Sumpter County Agricultural Center, Bushnell",
        "lat": 28.6616, "lng": -82.1027,
        "max_capacity": 550,
        "direction": "northeast",
        "is_inland": True,
        "has_medical": False,
        "notes": "Inland agricultural fairground 55 miles northeast on high ground",
    },
]


# ── Zone selection agent ───────────────────────────────────────────────────────

_zone_agent = LlmAgent(
    name="safe_zone_coordinator",
    description=(
        "Selects optimal hurricane evacuation safe zones for Tampa Bay by reasoning "
        "over a candidate pool. Calls hurricane_analysis_agent via Agent2Agent to get "
        "the threat assessment, then thinks through which candidates are safe given "
        "the storm trajectory, wind radius, surge zones, and civilian medical needs."
    ),
    model="gemini-2.5-flash",
    tools=[AgentTool(agent=hurricane_analysis_agent)],
    instruction="""You are Tampa Bay's Emergency Evacuation Zone Coordinator.
You have access to a list of REAL candidate safe zones and the hurricane_analysis_agent tool.

STEP 1 — CALL hurricane_analysis_agent
  Pass: category, wind_speed, eye_lat, eye_lng, distance_km
  Receive: wind_radius_km, storm_surge_ft, flooded_zones, safe_evacuation_directions,
           threat_level, hours_to_landfall, danger_zone_radius_km

STEP 2 — THINK through the candidate list carefully:
  For EACH candidate zone ask yourself:
  - Is it OUTSIDE the danger_zone_radius_km from the current eye position?
  - Is it NOT in a flooded zone or coastal area?
  - Is its compass direction one of the safe_evacuation_directions?
  - Does it have enough capacity for the threat_level?
  - If civilians have medical needs, is has_medical=true?
  - Is the zone actually reachable by road from Tampa Bay?

  Reject zones that fail any of these checks.
  Think about the storm's projected path — zones directly in the track are dangerous
  even if currently outside the wind radius.

STEP 3 — SELECT the best 5–7 zones from the surviving candidates:
  - Spread selections across multiple safe directions (don't cluster)
  - Include at least 2 zones with has_medical=true
  - Scale target capacity to threat_level:
      moderate → select zones totaling 1500+ capacity
      high     → 2000+ capacity
      severe   → 3000+ capacity
      extreme  → 4500+ capacity
  - Prefer higher-capacity zones for extreme/severe threats
  - Include at least one zone 50+ miles from Tampa for extreme threats

STEP 4 — Return ONLY valid JSON, no markdown:
{"safe_zones": [
  {"id": "<use candidate id>",
   "name": "<exact candidate name>",
   "lat": <exact candidate lat>,
   "lng": <exact candidate lng>,
   "capacity": <set based on threat_level, up to max_capacity>,
   "supplies": {"food": <capacity*1.5>, "water": <capacity*3>, "medical": <if has_medical: capacity*0.2, else capacity*0.05>},
   "reasoning": "2-3 sentences: why this zone is safe given the threat data, why it fits this storm's trajectory"}
]}""",
)

_runner = Runner(agent=_zone_agent, app_name="hurricane_evac", session_service=_session_service)


async def generate_safe_zones(origin_lng, origin_lat, dest_lng, dest_lat, category, wind_speed):
    control_points = build_control_points(origin_lng, origin_lat, dest_lng, dest_lat)
    path_points    = get_path_sample_points(control_points, steps=10)
    path_str       = " → ".join([f"({p['lat']:.2f}°N, {p['lng']:.2f}°W)" for p in path_points])

    distance_km = round(haversine_km(origin_lat, origin_lng, 27.9506, -82.4572))

    # Format the candidate pool for the AI
    candidate_lines = []
    for z in CANDIDATE_ZONES:
        dist_from_eye = round(haversine_km(origin_lat, origin_lng, z["lat"], z["lng"]))
        dist_from_tampa = round(haversine_km(27.9506, -82.4572, z["lat"], z["lng"]))
        candidate_lines.append(
            f"  [{z['id']}] {z['name']} | "
            f"lat={z['lat']}, lng={z['lng']} | "
            f"max_capacity={z['max_capacity']} | "
            f"direction={z['direction']} | "
            f"inland={z['is_inland']} | "
            f"has_medical={z['has_medical']} | "
            f"{dist_from_eye}km from eye | {dist_from_tampa}km from Tampa | "
            f"notes: {z['notes']}"
        )

    candidates_block = "\n".join(candidate_lines)

    prompt = (
        f"Hurricane Category {category}, wind speed {wind_speed} mph. "
        f"Eye is at ({origin_lat:.3f}°N, {origin_lng:.3f}°W), "
        f"{distance_km} km from Tampa Bay. "
        f"Projected path: {path_str}.\n\n"
        f"STEP 1: Call hurricane_analysis_agent with: "
        f"category={category}, wind_speed={wind_speed}, "
        f"eye_lat={origin_lat:.4f}, eye_lng={origin_lng:.4f}, distance_km={distance_km}.\n\n"
        f"STEP 2: Evaluate each of the following {len(CANDIDATE_ZONES)} REAL candidate zones. "
        f"Use the threat data to reason about which are safe for this specific storm:\n\n"
        f"{candidates_block}\n\n"
        f"STEP 3: Select the best 5-7 zones and return JSON."
    )

    session_id = str(uuid.uuid4())
    await _session_service.create_session(
        app_name="hurricane_evac", user_id="system", session_id=session_id
    )
    content = types.Content(role="user", parts=[types.Part(text=prompt)])

    response_text = ""
    async for event in _runner.run_async(
        user_id="system", session_id=session_id, new_message=content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            response_text = event.content.parts[0].text

    try:
        text  = re.sub(r"```json\s*|```\s*", "", response_text.strip()).strip()
        data  = json.loads(text)
        zones = data.get("safe_zones", [])
        # Validate: coordinates must be within Florida bounding box
        valid = [z for z in zones if 26.5 < z.get("lat", 0) < 30.5 and -87.5 < z.get("lng", 0) < -80.0]
        return valid if valid else _fallback_zones(category, origin_lat, origin_lng)
    except Exception:
        return _fallback_zones(category, origin_lat, origin_lng)


def _fallback_zones(category: int = 3, eye_lat: float = 24.5, eye_lng: float = -83.0):
    """
    Select from the real candidate pool when the agent call fails.
    Picks inland zones in safe directions (north/northeast/east) that are
    outside a rough danger radius for the given category.
    """
    danger_radius = {1: 111, 2: 148, 3: 185, 4: 222, 5: 278}.get(category, 185)
    scale    = max(1.0, category * 0.3)
    med_base = int(40 * scale)

    safe_directions = {"north", "northeast", "east"}

    # Filter: inland, safe direction, outside danger radius
    candidates = [
        z for z in CANDIDATE_ZONES
        if z["is_inland"]
        and z["direction"] in safe_directions
        and haversine_km(eye_lat, eye_lng, z["lat"], z["lng"]) > danger_radius
    ]

    # Sort: medical zones first, then by distance from eye (safest = farthest)
    candidates.sort(key=lambda z: (
        0 if z["has_medical"] else 1,
        -haversine_km(eye_lat, eye_lng, z["lat"], z["lng"])
    ))

    # Take up to 6; ensure at least 2 with medical if possible
    selected = []
    medical_count = 0
    for z in candidates:
        if len(selected) >= 6:
            break
        selected.append(z)
        if z["has_medical"]:
            medical_count += 1

    # Build output in expected format
    result = []
    for z in selected:
        cap = min(int(z["max_capacity"] * scale), z["max_capacity"])
        med = int(cap * 0.2) if z["has_medical"] else med_base
        result.append({
            "id":       z["id"],
            "name":     z["name"],
            "lat":      z["lat"],
            "lng":      z["lng"],
            "capacity": cap,
            "supplies": {"food": int(cap * 1.5), "water": int(cap * 3), "medical": med},
            "reasoning": z["notes"],
        })

    return result if result else _minimal_fallback(category)


def _minimal_fallback(category: int):
    """Last-resort hardcoded zones if the pool filter returns nothing."""
    scale = max(1.0, category * 0.3)
    return [
        {"id": "cz12", "name": "Strawberry Festival Grounds, Plant City",
         "lat": 28.0189, "lng": -82.1143,
         "capacity": int(500 * scale),
         "supplies": {"food": int(750 * scale), "water": int(1500 * scale), "medical": int(40 * scale)},
         "reasoning": "25 miles east of Tampa, inland, above surge elevation"},
        {"id": "cz13", "name": "RP Funding Center, Lakeland",
         "lat": 28.0395, "lng": -81.9498,
         "capacity": int(450 * scale),
         "supplies": {"food": int(675 * scale), "water": int(1350 * scale), "medical": int(50 * scale)},
         "reasoning": "35 miles east, Lakeland Regional Health nearby"},
        {"id": "cz04", "name": "Yuengling Center – USF, Tampa",
         "lat": 28.0655, "lng": -82.4149,
         "capacity": int(400 * scale),
         "supplies": {"food": int(600 * scale), "water": int(1200 * scale), "medical": int(40 * scale)},
         "reasoning": "North of downtown Tampa, university campus with medical resources"},
        {"id": "cz07", "name": "Hernando County Fairgrounds, Brooksville",
         "lat": 28.5575, "lng": -82.3832,
         "capacity": int(500 * scale),
         "supplies": {"food": int(750 * scale), "water": int(1500 * scale), "medical": int(30 * scale)},
         "reasoning": "40 miles north on high inland terrain"},
        {"id": "cz11", "name": "University of Florida – O'Connell Center, Gainesville",
         "lat": 29.6516, "lng": -82.3486,
         "capacity": int(600 * scale),
         "supplies": {"food": int(900 * scale), "water": int(1800 * scale), "medical": int(80 * scale)},
         "reasoning": "90 miles northeast, UF Shands hospital on-site"},
    ]
