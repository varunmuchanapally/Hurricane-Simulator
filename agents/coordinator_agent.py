import json, re, uuid
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

_agent = LlmAgent(
    name="coordinator_agent",
    model="gemini-2.5-flash",
    description="Tampa Bay hurricane evacuation coordinator providing real-time situational assessment",
    instruction="""You are the Tampa Bay hurricane evacuation coordinator.
Assess the current evacuation status and provide actionable guidance.
Return ONLY valid JSON, no markdown:
{"overall_status": "on_track|at_risk|critical", "alerts": ["alert1"], "recommendations": ["action1"], "broadcast": "max 15 word message to all agents"}""",
)

_session_service = InMemorySessionService()
_runner = Runner(agent=_agent, app_name="hurricane_evac", session_service=_session_service)


async def get_coordinator_assessment(agents, safe_zones, hurricane, elapsed_hours):
    total      = len(agents)
    arrived    = sum(1 for a in agents if a["status"] == "arrived")
    evacuating = sum(1 for a in agents if a["status"] == "evacuating")
    stranded   = sum(1 for a in agents if a["status"] == "stranded")

    zone_str = "\n".join([
        f"- {z['name']}: "
        f"{sum(1 for a in agents if a.get('assignedZoneId') == z['id'] and a['status'] == 'arrived')} arrived, "
        f"{round((sum(1 for a in agents if a.get('assignedZoneId') == z['id'] and a['status'] == 'arrived') / z['capacity']) * 100)}% full"
        for z in safe_zones
    ])

    prompt = (
        f"After {elapsed_hours:.1f} hours — {arrived}/{total} safe, {evacuating} evacuating, {stranded} stranded. "
        f"Hurricane Category {hurricane.get('category', '?')}, {hurricane.get('windSpeed', '?')} mph. "
        f"Zone status:\n{zone_str}"
    )

    session_id = str(uuid.uuid4())
    await _session_service.create_session(app_name="hurricane_evac", user_id="system", session_id=session_id)
    content = types.Content(role="user", parts=[types.Part(text=prompt)])

    response_text = ""
    async for event in _runner.run_async(user_id="system", session_id=session_id, new_message=content):
        if event.is_final_response() and event.content and event.content.parts:
            response_text = event.content.parts[0].text

    try:
        text = re.sub(r"```json\s*|```\s*", "", response_text.strip()).strip()
        return json.loads(text)
    except Exception:
        return {"overall_status": "on_track", "alerts": [], "recommendations": ["Continue evacuation"], "broadcast": "Evacuation proceeding. Follow assigned routes."}
