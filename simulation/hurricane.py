import math

TAMPA_CENTER = {"lng": -82.4572, "lat": 27.9506}

def catmull_rom(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2 * p1)
        + (-p0 + p2) * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
    )

def build_control_points(origin_lng, origin_lat, dest_lng, dest_lat):
    tc = TAMPA_CENTER
    p0 = {"lng": origin_lng + (origin_lng - tc["lng"]) * 0.5, "lat": origin_lat + (origin_lat - tc["lat"]) * 0.5}
    p1 = {"lng": origin_lng, "lat": origin_lat}
    p2 = {"lng": tc["lng"],  "lat": tc["lat"]}
    p3 = {"lng": dest_lng,   "lat": dest_lat}
    p4 = {"lng": dest_lng + (dest_lng - tc["lng"]) * 0.5, "lat": dest_lat + (dest_lat - tc["lat"]) * 0.5}
    return [p0, p1, p2, p3, p4]

def interpolate_position(control_points, t):
    segments = len(control_points) - 3
    scaled = t * segments
    seg_idx = min(int(scaled), segments - 1)
    seg_t = scaled - seg_idx
    p0 = control_points[seg_idx]
    p1 = control_points[seg_idx + 1]
    p2 = control_points[seg_idx + 2]
    p3 = control_points[seg_idx + 3]
    return {
        "lng": catmull_rom(p0["lng"], p1["lng"], p2["lng"], p3["lng"], seg_t),
        "lat": catmull_rom(p0["lat"], p1["lat"], p2["lat"], p3["lat"], seg_t),
    }

def haversine_km(lat1, lng1, lat2, lng2) -> float:
    R = 6371
    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)
    a = (math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lng / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def distance_from_path(point_lat, point_lng, control_points, steps=50) -> float:
    min_dist = float("inf")
    for i in range(steps + 1):
        pos = interpolate_position(control_points, i / steps)
        d = haversine_km(point_lat, point_lng, pos["lat"], pos["lng"])
        if d < min_dist:
            min_dist = d
    return min_dist

def get_path_sample_points(control_points, steps=20):
    points = []
    for i in range(steps + 1):
        pos = interpolate_position(control_points, i / steps)
        points.append({"lat": round(pos["lat"], 3), "lng": round(pos["lng"], 3)})
    return points