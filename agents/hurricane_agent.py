"""
Hurricane Analysis Agent — the meteorological brain of the system.

This is Core Agent 1.  It is called by the Safe Zone Agent via Agent2Agent
(AgentTool) before zone selection.  It uses deterministic physics tools so
the LLM reads real computed values — not hallucinated numbers.

Physics tools:
  get_wind_radius(category)    — Saffir-Simpson wind radii (NHC averages)
  get_storm_surge_ft(category) — NOAA surge tables, Tampa Bay amplified

The same threat data is also stored in _hurricane_state so the 80 civilian
agents can read it from their lightweight get_hurricane_status() tool without
each making an extra A2A call.
"""

from pydantic import BaseModel
from google.adk.agents import LlmAgent


# ── Structured input schema (enables typed A2A via AgentTool) ─────────────────
class HurricaneAnalysisInput(BaseModel):
    category:    int
    wind_speed:  float
    eye_lat:     float
    eye_lng:     float
    distance_km: float   # current distance of eye from Tampa Bay


# ── Physics tools ─────────────────────────────────────────────────────────────
# Passed as tools=[...] to the LLM agent.  The model calls them; it does NOT
# hallucinate the numbers.

def get_wind_radius(category: int) -> dict:
    """
    Return the radius in km of tropical-storm-force winds (≥39 mph) and the
    inner hurricane-force core (≥74 mph) for this hurricane category.
    Based on National Hurricane Center Saffir-Simpson scale averages for
    Gulf Coast storms.
    """
    ts_radii  = {1: 111, 2: 148, 3: 185, 4: 222, 5: 278}   # tropical-storm force
    hf_radii  = {1:  50, 2:  67, 3:  83, 4: 100, 5: 125}   # hurricane force
    cat = max(1, min(5, category))
    return {
        "wind_radius_km":           ts_radii[cat],
        "hurricane_force_radius_km": hf_radii[cat],
        "note": (
            f"Category {cat}: tropical-storm winds extend {ts_radii[cat]} km, "
            f"hurricane-force winds extend {hf_radii[cat]} km from eye"
        ),
    }


def get_storm_surge_ft(category: int) -> dict:
    """
    Return expected storm surge height in feet at Tampa Bay coastline for this
    category.  Tampa Bay is especially vulnerable — the funnel shape of the bay
    amplifies surge by ~20 percent over open-coast values.
    Based on NOAA storm surge forecasting guidelines (NHC SLOSH model averages).
    """
    base_surge = {1: 5, 2: 8, 3: 12, 4: 18, 5: 25}
    cat        = max(1, min(5, category))
    tampa_ft   = round(base_surge[cat] * 1.2)   # bay amplification

    flooded = []
    if tampa_ft >= 3:  flooded.append("Bayshore Boulevard, Davis Islands")
    if tampa_ft >= 6:  flooded.append("Hyde Park, Harbour Island, Ybor City waterfront")
    if tampa_ft >= 9:  flooded.append("Downtown Tampa, St. Pete Beach, Pinellas coast")
    if tampa_ft >= 13: flooded.append("Virtually all South Tampa, coastal Hillsborough County")
    if tampa_ft >= 18: flooded.append("Catastrophic inundation — all bay-adjacent areas")

    return {
        "storm_surge_ft":   tampa_ft,
        "flooded_zones":    flooded,
        "note": (
            f"Category {cat} surge at Tampa Bay: up to {tampa_ft} ft "
            f"(20% bay-amplification applied to {base_surge[cat]} ft open-coast value)"
        ),
    }


# ── Hurricane Analysis Agent ──────────────────────────────────────────────────

hurricane_analysis_agent = LlmAgent(
    name="hurricane_analysis_agent",
    description=(
        "Analyzes hurricane parameters and returns a structured threat assessment "
        "for Tampa Bay evacuation planning.  Calls physics tools to compute exact "
        "wind radius and storm surge — does not guess.  Called by the Safe Zone Agent "
        "via Agent2Agent before zone selection."
    ),
    model="gemini-2.5-flash",
    input_schema=HurricaneAnalysisInput,
    tools=[get_wind_radius, get_storm_surge_ft],
    instruction="""You are the Tampa Bay Emergency Management meteorological analysis system.

You receive structured hurricane parameters from the Safe Zone Agent (Agent2Agent call).

YOUR STEPS:
1. Call get_wind_radius(category) — get the exact km radius of damaging winds
2. Call get_storm_surge_ft(category) — get coastal flood heights and affected zones
3. Compute hours_to_landfall: distance_km / 25  (25 km/h = typical Gulf forward speed)
4. Determine safe evacuation directions based on storm approach angle:
   - Storm from southwest → safe corridors are NORTH and NORTHEAST
   - Storm from west     → safe corridors are NORTH and EAST
   - Storm from south    → safe corridors are NORTH, NORTHEAST, NORTHWEST
   - Never direct evacuees toward the storm track
5. Set threat_level:
   - "extreme"  : Category 4-5 AND distance_km < 200
   - "severe"   : Category 3+  OR  distance_km < 150
   - "high"     : Category 2+  OR  distance_km < 300
   - "moderate" : Category 1, distance_km ≥ 300

Return ONLY valid JSON, no markdown:
{
  "wind_radius_km": <number>,
  "hurricane_force_radius_km": <number>,
  "storm_surge_ft": <number>,
  "flooded_zones": ["zone description"],
  "hours_to_landfall": <number>,
  "threat_level": "extreme|severe|high|moderate",
  "danger_zone_radius_km": <number>,
  "safe_evacuation_directions": ["north", "northeast"],
  "coastal_areas_at_risk": ["area names"],
  "assessment": "Two-sentence threat summary for the emergency briefing."
}""",
)
