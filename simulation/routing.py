import asyncio
from agents.route_agent import (
    assign_zones_via_agent,
    run_civilian_agent,
    set_hurricane_state,
    _route_cache,
    _haversine_km,
)


def _straight_route(from_lng, from_lat, to_lng, to_lat, steps=12):
    return [
        [from_lng + (to_lng - from_lng) * i / steps,
         from_lat + (to_lat - from_lat) * i / steps]
        for i in range(steps + 1)
    ]


async def assign_agents_to_zones(agents: list, safe_zones: list, hurricane: dict = None) -> list:
    """
    Full AI-driven evacuation routing pipeline:

      Phase 1 — Zone Agent assigns each civilian to the best safe zone
                 (medical needs, capacity, distance, situation).

      Phase 2 — Each civilian runs as its own ADK agent, calls
                 get_hurricane_status() and get_routes_to_safe_zone() as
                 external tools, then decides action + which road to take.

      Phase 3 — Assemble final assignment objects with coordinates, distances,
                 and ETAs from the Mapbox route the agent chose.
    """
    zone_map = {z["id"]: z for z in safe_zones}

    # ── Seed hurricane state so the tool returns live data ────────────────────
    if hurricane:
        set_hurricane_state(hurricane)

    # ── Phase 1: Zone assignment ──────────────────────────────────────────────
    zone_assignments = await assign_zones_via_agent(agents, safe_zones)
    zone_lookup = {a["agent_id"]: a for a in zone_assignments}

    # ── Phase 2: Each civilian is an ADK agent with external tool calls ───────
    # Build (civilian, zone) pairs for agents that have a valid zone assignment
    tasks = []
    for agent in agents:
        zone_asgn = zone_lookup.get(agent["id"], {})
        zone_id   = zone_asgn.get("zone_id")
        zone      = zone_map.get(zone_id) if zone_id else None
        tasks.append((agent, zone))

    # Run all 80 civilian agents in parallel
    decisions = await asyncio.gather(*[
        run_civilian_agent(agent, zone) if zone else _no_zone_decision(agent)
        for agent, zone in tasks
    ])
    decision_map = {d["agent_id"]: d for d in decisions}

    # ── Phase 3: Build final results ──────────────────────────────────────────
    results = []
    for agent in agents:
        agent_id  = agent["id"]
        zone_asgn = zone_lookup.get(agent_id, {})
        zone_id   = zone_asgn.get("zone_id")
        zone      = zone_map.get(zone_id) if zone_id else None

        if not zone:
            results.append({
                "agent_id":        agent_id,
                "zone_id":         None,
                "route":           None,
                "distance_km":     None,
                "eta_hours":       None,
                "status":          "stranded",
                "zone_reasoning":  zone_asgn.get("reasoning", ""),
                "route_reasoning": "No zone available.",
            })
            continue

        # Get the route the civilian agent chose
        decision  = decision_map.get(agent_id, {})
        route_idx = decision.get("route_index", 0)
        fetched   = _route_cache.get(agent_id, [])
        chosen    = next((r for r in fetched if r["index"] == route_idx), None)

        if chosen and chosen.get("coords"):
            coords  = chosen["coords"]
            dist_km = chosen["distance_km"]
        else:
            # Mapbox unavailable — straight line fallback
            pos     = agent["position"]
            coords  = _straight_route(pos["lng"], pos["lat"], zone["lng"], zone["lat"])
            dist_km = _haversine_km(pos["lat"], pos["lng"], zone["lat"], zone["lng"])

        speed = agent.get("speed", 0.7)
        results.append({
            "agent_id":        agent_id,
            "zone_id":         zone_id,
            "route":           coords,
            "distance_km":     round(dist_km, 2),
            "eta_hours":       round(dist_km / (speed * 80), 2),
            "status":          "waiting",
            "zone_reasoning":  zone_asgn.get("reasoning", ""),
            "route_reasoning": decision.get("reasoning", ""),
        })

    return results


async def _no_zone_decision(agent: dict) -> dict:
    return {
        "agent_id":    agent["id"],
        "action":      "request_help",
        "route_index": 0,
        "reasoning":   "No safe zone was assigned.",
        "message":     None,
    }
