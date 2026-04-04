import json, re, uuid
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from simulation.hurricane import haversine_km

_agent = LlmAgent(
    name="civilian_agent",
    model="gemini-2.5-flash",
    description="Reasons about individual civilian evacuation decisions during a Tampa Bay hurricane",
    instruction="""You simulate real people facing a hurricane evacuation order in Tampa Bay, Florida.
Each person has a unique name, age, background, personal situation, and personality.
Your job is to reason about what each person would REALISTICALLY do — not what they should do.
Some people will refuse to leave. Some cannot leave. Some will comply immediately.
This is a simulation of human behavior under stress — be authentic, not idealized.
Return ONLY the requested JSON. No markdown, no explanation.""",
)

_session_service = InMemorySessionService()
_runner = Runner(agent=_agent, app_name="hurricane_evac", session_service=_session_service)


async def get_all_decisions_batch(agents, hurricane, safe_zones):
    """Single LLM call reasoning about every agent — returns a decision array."""
    zone_map = {z["id"]: z for z in safe_zones}

    lines = []
    for a in agents:
        zone = zone_map.get(a.get("assignedZoneId") or a.get("assigned_zone_id"))
        if zone:
            dist = round(haversine_km(
                a["position"]["lat"], a["position"]["lng"],
                zone["lat"], zone["lng"],
            ))
            zone_info = f"{zone['name']} ({dist} km away)"
        else:
            zone_info = "no zone assigned"

        lines.append(
            f"ID:{a['id']} | {a.get('name', 'Unknown')} | {a.get('type', 'adult')}, "
            f"age {a.get('age', '?')} | {a.get('neighborhood', 'Tampa')} | "
            f"Situation: {a.get('situation', 'N/A')} | "
            f"Personality: {a.get('personality', 'rational')} | "
            f"Medical needs: {'yes' if a.get('needsMedical') or a.get('needs_medical') else 'no'} | "
            f"Assigned evac zone: {zone_info}"
        )

    agents_block = "\n".join(lines)

    prompt = (
        f"Hurricane Category {hurricane.get('category', '?')} "
        f"({hurricane.get('windSpeed', '?')} mph) is approaching Tampa Bay. "
        f"A mandatory evacuation order is now in effect.\n\n"
        f"Below are {len(agents)} real people in the affected area. "
        f"Reason about each one individually based on their personal profile and circumstances. "
        f"Be realistic — not everyone evacuates even when ordered to. "
        f"Personality and situation should heavily influence each decision.\n\n"
        f"{agents_block}\n\n"
        f"Return a JSON array (no markdown, no other text) with one entry per agent:\n"
        f'[{{"agent_id":"...","action":"evacuate|shelter_in_place|request_help",'
        f'"urgency":"low|medium|high|critical",'
        f'"reasoning":"one sentence in their voice explaining their choice",'
        f'"message":"brief radio broadcast or null"}}]'
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
        text = re.sub(r"```json\s*|```\s*", "", response_text.strip()).strip()
        return json.loads(text)
    except Exception:
        return [
            {"agent_id": a["id"], "action": "evacuate", "urgency": "medium",
             "reasoning": "Following the official evacuation order.", "message": None}
            for a in agents
        ]


async def get_batch_decisions(agents, hurricane, safe_zones):
    """Called by /api/start-simulation — delegates to batch function."""
    return await get_all_decisions_batch(agents, hurricane, safe_zones)
