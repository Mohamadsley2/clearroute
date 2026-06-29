"""
routing.py — ClearRoute: collection route planning.

Hybrid pipeline (AI + Google Maps):
  1. geocode_depot()      -> depot coordinates (Google Geocoding, cached)
  2. plan_assignment()    -> Claude (claude-opus-4-8) assigns points to vehicles and
                            orders each route; if there is no key or it fails, a local
                            fallback runs (k-means + nearest neighbour)
  3. directions_for_route -> Google Directions: real road path + distance/time

This module has NO Streamlit dependency (pure logic). Keys are passed in as
arguments from app.py.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache

import requests

# AI model (do not change without reason: claude-opus-4-8 is the most capable).
CLAUDE_MODEL = "claude-opus-4-8"

# Depot: Fritz-Arnold-Straße 2B, 78467 Konstanz-Industriegebiet.
DEPOT_ADDRESS = "Fritz-Arnold-Straße 2B, 78467 Konstanz, Germany"
# Fallback when there is no Google key (real coords of the address, via Geocoding).
DEPOT_FALLBACK = (47.67709, 9.14396)

# Above this count we pre-group with k-means before sending to Claude (zone hint).
CLUSTER_HINT_THRESHOLD = 150

# Google Directions allows at most 25 locations per request
# (origin + destination + 23 waypoints). We chunk below that limit.
MAX_WAYPOINTS = 23

# Colour palette per vehicle (repeats if there are more vehicles than colours).
PALETTE = [
    "#FF4B4B", "#1E88E5", "#2ECC71", "#FF8C00", "#8E44AD",
    "#00BCD4", "#FFC300", "#E91E63", "#795548", "#607D8B",
]


# ── Geographic helpers ─────────────────────────────────────────────────────────
def haversine_m(a, b):
    """Distance in metres between (lat, lon) points a and b."""
    r = 6371000.0
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def color_for(vehicle_index):
    return PALETTE[vehicle_index % len(PALETTE)]


# ── 1) Depot geocoding ──────────────────────────────────────────────────────────
@lru_cache(maxsize=32)
def geocode_depot(address, api_key):
    """
    Returns (lat, lon) of the depot via the Google Geocoding API.
    If there is no key or the call fails, returns DEPOT_FALLBACK.
    Cached by (address, api_key).
    """
    if not api_key:
        return DEPOT_FALLBACK
    try:
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"address": address, "key": api_key},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "OK" and data.get("results"):
            loc = data["results"][0]["geometry"]["location"]
            return (loc["lat"], loc["lng"])
    except Exception:
        pass
    return DEPOT_FALLBACK


# ── 2a) Clustering (zone hint / assignment fallback) ────────────────────────────
def cluster_points(points, k):
    """
    Assigns each point a zone label (0..k-1) using k-means on (lat, lon).
    If scikit-learn is unavailable, uses a simple split by longitude.
    """
    k = max(1, min(k, len(points)))
    coords = [(p["lat"], p["lon"]) for p in points]
    try:
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        return list(model.fit_predict(coords))
    except Exception:
        # Fallback without sklearn: sort by longitude and split into k bands.
        order = sorted(range(len(points)), key=lambda i: coords[i][1])
        labels = [0] * len(points)
        per = math.ceil(len(points) / k)
        for rank, idx in enumerate(order):
            labels[idx] = min(rank // per, k - 1)
        return labels


# ── 2b) Assignment + ordering with Claude ───────────────────────────────────────
ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "routes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "vehicle": {"type": "integer"},
                    "stop_ids": {"type": "array", "items": {"type": "integer"}},
                    "reasoning": {"type": "string"},
                },
                "required": ["vehicle", "stop_ids", "reasoning"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["routes"],
    "additionalProperties": False,
}


def _claude_assignment(points, n_vehicles, depot, api_key, model=CLAUDE_MODEL):
    """
    Calls Claude to assign points to vehicles and order each route.
    Returns [{vehicle, stop_ids, reasoning}] or raises if it fails / is invalid.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    hints = None
    if len(points) > CLUSTER_HINT_THRESHOLD:
        hints = cluster_points(points, max(n_vehicles, min(8, len(points))))

    compact = []
    for i, p in enumerate(points):
        row = {
            "id": p["id"],
            "lat": round(p["lat"], 5),
            "lon": round(p["lon"], 5),
            "typ": p["typ"],
            "score": round(p.get("score", 0.0), 2),
        }
        if hints is not None:
            row["zone"] = int(hints[i])
        compact.append(row)

    system = (
        "You are a logistics planner for the urban cleaning service of Konstanz. "
        "You solve a vehicle routing problem (VRP): assign every litter point to exactly "
        "one vehicle and order each vehicle's stops into an efficient route that starts and "
        "ends at the depot.\n"
        "Criteria, in order: (1) every point is assigned to ONE vehicle and none is left "
        "unassigned; (2) balance the number of stops across vehicles; (3) group by "
        "geographic proximity to minimise travel; (4) within each route, prioritise the "
        "highest-'score' (most urgent) points when it does not penalise distance much. "
        "Return ONLY the JSON described by the schema."
    )

    user = (
        f"Depot (start and end of all routes): lat={depot[0]:.5f}, lon={depot[1]:.5f}.\n"
        f"Vehicles available: {n_vehicles}.\n"
        f"Detected litter points ({len(points)}):\n"
        f"{json.dumps(compact, ensure_ascii=False)}\n\n"
        "Return an object with 'routes': one entry per vehicle (vehicle = 1..N), "
        "'stop_ids' with the ids IN VISIT ORDER, and 'reasoning' (1-2 sentences in English). "
        "Use the ids exactly as they appear above. Do not invent ids. Do not repeat ids."
    )

    with client.messages.stream(
        model=model,
        max_tokens=16000,
        system=system,
        output_config={
            "format": {"type": "json_schema", "schema": ROUTE_SCHEMA},
            "effort": "high",
        },
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": user}],
    ) as stream:
        msg = stream.get_final_message()

    if msg.stop_reason == "refusal":
        raise RuntimeError("Claude refused the request (stop_reason=refusal).")

    text = next((b.text for b in msg.content if b.type == "text"), None)
    if not text:
        raise RuntimeError("Claude returned no JSON text.")

    data = json.loads(text)
    routes = data["routes"]

    # Validation: every id exactly once.
    all_ids = {p["id"] for p in points}
    assigned = [sid for r in routes for sid in r["stop_ids"]]
    if sorted(assigned) != sorted(all_ids):
        raise RuntimeError("Invalid assignment from Claude (missing or duplicated ids).")

    return routes


def _fallback_assignment(points, n_vehicles, depot):
    """
    Deterministic assignment without AI: k-means into n_vehicles zones + nearest-neighbour
    ordering from the depot. Used when there is no key or Claude fails.
    """
    by_id = {p["id"]: p for p in points}
    labels = cluster_points(points, n_vehicles)
    groups = {}
    for p, lab in zip(points, labels):
        groups.setdefault(lab, []).append(p["id"])

    routes = []
    for v, (_, ids) in enumerate(sorted(groups.items()), start=1):
        # Nearest neighbour from the depot.
        remaining = list(ids)
        ordered = []
        current = depot
        while remaining:
            nxt = min(remaining, key=lambda sid: haversine_m(current, (by_id[sid]["lat"], by_id[sid]["lon"])))
            ordered.append(nxt)
            current = (by_id[nxt]["lat"], by_id[nxt]["lon"])
            remaining.remove(nxt)
        routes.append({
            "vehicle": v,
            "stop_ids": ordered,
            "reasoning": "Zone-based assignment (k-means) with nearest-neighbour ordering.",
        })
    return routes


# ── 3) Google Directions (real road path) ───────────────────────────────────────
def _latlng(p):
    return f"{p['lat']:.6f},{p['lon']:.6f}"


@lru_cache(maxsize=256)
def _directions_request(origin, destination, waypoints, api_key):
    """One Directions request. waypoints = 'lat,lng|lat,lng' string or ''."""
    params = {
        "origin": origin,
        "destination": destination,
        "key": api_key,
        "departure_time": "now",  # traffic-aware ETA when the account allows it
    }
    if waypoints:
        params["waypoints"] = waypoints
    resp = requests.get(
        "https://maps.googleapis.com/maps/api/directions/json",
        params=params,
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()


def directions_for_route(depot, ordered_points, api_key):
    """
    Computes the real path depot -> stops (in order) -> depot.
    Chunks if there are more than MAX_WAYPOINTS intermediate stops.
    Returns {"polylines":[enc,...], "distance_m":int, "duration_s":int, "ok":bool}.
    """
    if not api_key or not ordered_points:
        return {"polylines": [], "distance_m": 0, "duration_s": 0, "ok": False}

    depot_str = f"{depot[0]:.6f},{depot[1]:.6f}"
    seq = [depot_str] + [_latlng(p) for p in ordered_points] + [depot_str]

    polylines, distance_m, duration_s = [], 0, 0
    i = 0
    try:
        while i < len(seq) - 1:
            j = min(i + 1 + MAX_WAYPOINTS, len(seq) - 1)
            origin, destination = seq[i], seq[j]
            waypoints = "|".join(seq[i + 1:j])
            data = _directions_request(origin, destination, waypoints, api_key)
            if data.get("status") != "OK" or not data.get("routes"):
                return {"polylines": polylines, "distance_m": distance_m,
                        "duration_s": duration_s, "ok": False}
            route = data["routes"][0]
            polylines.append(route["overview_polyline"]["points"])
            for leg in route["legs"]:
                distance_m += leg["distance"]["value"]
                duration_s += leg.get("duration_in_traffic", leg["duration"])["value"]
            i = j
        return {"polylines": polylines, "distance_m": distance_m,
                "duration_s": duration_s, "ok": True}
    except Exception:
        return {"polylines": polylines, "distance_m": distance_m,
                "duration_s": duration_s, "ok": False}


# ── Orchestrator ────────────────────────────────────────────────────────────────
def plan_routes(points, n_vehicles, depot, anthropic_key=None, gmaps_key=None,
                model=CLAUDE_MODEL):
    """
    Orchestrates the full pipeline and returns (routes, info).

    points: [{id, lat, lon, typ, konfidenz, score}]
    routes: [{vehicle, color, reasoning, stops:[{...,seq}], polylines, distance_m,
              duration_s, directions_ok}]
    info:   {"engine": "claude"|"fallback", "warnings": [..]}
    """
    info = {"engine": "claude", "warnings": []}
    n_vehicles = max(1, min(n_vehicles, len(points))) if points else n_vehicles

    if not points:
        return [], {"engine": "none", "warnings": ["No points to plan."]}

    # Assignment + ordering.
    assignment = None
    if anthropic_key:
        try:
            assignment = _claude_assignment(points, n_vehicles, depot, anthropic_key, model)
        except Exception as e:  # noqa: BLE001
            info["warnings"].append(f"Claude unavailable ({e}); using local assignment.")
    else:
        info["warnings"].append("No ANTHROPIC_API_KEY; using local assignment (no AI).")

    if assignment is None:
        assignment = _fallback_assignment(points, n_vehicles, depot)
        info["engine"] = "fallback"

    by_id = {p["id"]: p for p in points}
    routes = []
    for idx, r in enumerate(assignment):
        ordered = [dict(by_id[sid], seq=k + 1) for k, sid in enumerate(r["stop_ids"]) if sid in by_id]
        dirs = directions_for_route(depot, ordered, gmaps_key)
        if not dirs["ok"] and gmaps_key:
            info["warnings"].append(f"Directions failed for vehicle {r['vehicle']}; drawing a straight line.")
        routes.append({
            "vehicle": r["vehicle"],
            "color": color_for(idx),
            "reasoning": r.get("reasoning", ""),
            "stops": ordered,
            "polylines": dirs["polylines"],
            "distance_m": dirs["distance_m"],
            "duration_s": dirs["duration_s"],
            "directions_ok": dirs["ok"],
        })

    if not gmaps_key:
        info["warnings"].append("No GOOGLE_MAPS_API_KEY; the map and real distances are unavailable.")

    return routes, info
