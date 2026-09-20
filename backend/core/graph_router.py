"""
graph_router.py — Multimodal route planning.

PRIMARY SEARCH ALGORITHM: A* (A-star)
--------------------------------------
InsureRoute's official route-search algorithm is A*, evaluated as:

    f(n) = g(n) + h(n)

    g(n) = the real, accumulated edge weight from the origin to node n
           (identical to what Dijkstra would accumulate — no shortcuts).
    h(n) = an admissible estimate of the remaining cost from n to the
           destination, used only to decide which frontier node to expand
           next.

A* with an admissible heuristic is guaranteed to return the same optimal
path Dijkstra would. With a consistent heuristic it also never expands
more nodes than Dijkstra (up to tie-breaking), because h(n) lets it
prioritise nodes "in the direction of" the destination. HOW MANY nodes it
saves is graph-dependent: on the small 14-node demo topology the saving
is marginal and is zero for the headline Pune->Mumbai request (measured,
not assumed — see AIML_PROJECT_REPORT.md section 6.9).

Three routing objectives are supported (speed / cost / safety), each with
its OWN edge-weight attribute and its OWN heuristic — see
`astar_heuristic()` for the admissibility argument for each one. Where a
geographically-derived heuristic cannot be proven to never overestimate
the true remaining cost (the "safety" objective), h(n) = 0 is used
instead of guessing — at h(n) = 0, A* is mathematically identical to
Dijkstra, so correctness is never sacrificed for a plausible-looking but
unprovable heuristic.

`nx.astar_path` (like Dijkstra) only ever returns ONE optimal path per
call. The dashboard needs several ranked alternatives, so
`find_k_routes_astar()` wraps repeated A* calls in the same Yen's-
algorithm bookkeeping loop NetworkX's own `shortest_simple_paths` uses
internally — the only change is that the inner single-path subroutine is
our A* implementation instead of Dijkstra. A* is still doing 100% of the
actual pathfinding; nothing here claims `nx.astar_path` itself produces
k-shortest paths.
"""

import networkx as nx
import json
import logging
import math
import os
import random
from datetime import datetime
from typing import Callable, Optional

logger = logging.getLogger("insureroute.routing")

EARTH_RADIUS_KM = 6371.0

# Edge attribute used for each routing objective — UNCHANGED from before
# this refactor. This mapping is the single source of truth for both the
# A* search itself and the heuristic units (minutes for "speed", INR for
# "cost", the placeholder risk unit for "safety").
WEIGHT_ATTR_BY_PRIORITY = {
    "speed": "base_time_min",
    "cost": "cost_inr",
    "safety": "risk_weight",
}

# A deliberately generous upper bound on how fast cargo can conceivably
# move (roughly commercial air-freight cruise speed). This is NOT derived
# from the fastest edge currently present in graph_topology.json — doing
# that would make the heuristic inadmissible the instant a faster
# edge/mode is added to the topology. Being generous costs some pruning
# efficiency but guarantees the "speed" heuristic can never overestimate
# remaining travel time, which is required for A* to still guarantee an
# optimal route.
MAX_PLAUSIBLE_SPEED_KMH = 900.0

_modal_config_cache: Optional[dict] = None


def _load_modal_config() -> dict:
    global _modal_config_cache
    if _modal_config_cache is None:
        config_path = os.path.join(os.path.dirname(__file__), "..", "data", "multimodal_config.json")
        with open(config_path, encoding="utf-8") as f:
            _modal_config_cache = json.load(f)
    return _modal_config_cache


def _min_cost_per_km_inr() -> float:
    """Cheapest INR/km rate across all transport modes (currently sea freight).
    Used as a lower bound for the 'cost' heuristic — see astar_heuristic()."""
    modes = _load_modal_config()["modes"]
    return min(m["cost_per_km_inr"] for m in modes.values())


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two GPS points (Haversine formula).

    This is the shortest possible distance between two points on the
    Earth's surface, so it lower-bounds the length of any real
    road/rail/sea/air path between them (triangle inequality on a sphere).
    The heuristics below build on that — but see astar_heuristic() for the
    per-edge premise the topology DATA must also satisfy.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def astar_heuristic(node_id: str, destination_id: str, node_coords: dict, priority: str) -> float:
    """
    h(n): an estimate of the remaining cost from `node_id` to
    `destination_id`, in the same units as the edge weight used for
    `priority` (minutes for "speed", INR for "cost").

    ADMISSIBILITY (why each non-zero branch never overestimates the true cost):

    Both non-zero branches have the form h(n) = c * great_circle_km(n, dest).
    The proof needs exactly ONE per-edge premise, for every edge (u, v):

            weight(u, v)  >=  c * great_circle_km(u, v)              (P)

    Given (P), the triangle inequality for great-circle distance gives
    h(n) <= true remaining cost (admissible) and h(u) - h(v) <= weight(u, v)
    (consistent).

    (P) is NOT guaranteed by physics alone in this project: the hand-entered
    `distance_km` in graph_topology.json is *smaller* than the great-circle
    distance on 6 of 18 edges (e.g. PUNE_DEPOT->CP03 declares 42 km vs
    ~46.7 km great-circle). The heuristics are safe on today's data only
    because of deliberate slack in `c` — and that is enforced by tests
    (HeuristicPremiseTests), not assumed. If topology data is edited so (P)
    fails, those tests fail.

    * priority == "speed" (unit: minutes)
        c = 60 / MAX_PLAUSIBLE_SPEED_KMH  (min per km)
        (P) holds because the fastest edge in the topology implies 200 km/h
        (the air leg), far below the 900 km/h used here.

    * priority == "cost" (unit: INR)
        c = cheapest INR/km rate in multimodal_config.json (sea, Rs 8/km)
        (P) holds because every edge actually present is charged at
        Rs 18/km or more (rail 18, road 35, air 220).

    * priority == "safety" (unit: the project's risk_weight)
        h(n) = 0.
        `risk_weight` is NOT a spatial quantity — a short edge can carry
        more risk than a long one (e.g. a flooded 5 km stretch vs a
        clear 50 km highway) — so no function of geographic distance can
        be proven to never overestimate remaining risk. Rather than use
        an unprovable heuristic while still claiming optimal routes,
        h(n) = 0 is used, which makes A* mathematically identical to
        Dijkstra for this objective: still correct, just without the
        geographic speed-up.

    Consistency (monotonicity) means A* never has to re-expand an
    already-explored node. Both properties are verified empirically for
    every destination and several cargo variants in test_graph_router.py.
    """
    if node_id == destination_id:
        return 0.0
    if priority == "safety":
        return 0.0
    if node_id not in node_coords or destination_id not in node_coords:
        return 0.0

    n1, n2 = node_coords[node_id], node_coords[destination_id]
    km = haversine_distance(n1["lat"], n1["lon"], n2["lat"], n2["lon"])

    if priority == "cost":
        return km * _min_cost_per_km_inr()

    # default / "speed"
    return (km / MAX_PLAUSIBLE_SPEED_KMH) * 60.0


def build_constrained_graph(topology_path: str,
                             blocked_checkpoints: list = None,
                             cargo_weight_tons: float = 1.0,
                             cargo_type: str = "fmcg") -> nx.Graph:
    """Builds the routing graph with every hard constraint already applied,
    BEFORE any search runs — blocked/high-risk checkpoints are removed as
    nodes, and cargo-incompatible legs (overweight, hazmat-on-rail, cold
    chain unavailable) are simply never added as edges. A* (or any other
    algorithm) therefore only ever sees valid states; nothing is filtered
    out after the fact."""
    if blocked_checkpoints is None:
        blocked_checkpoints = []

    G = nx.Graph()
    with open(topology_path, encoding="utf-8") as f:
        topo = json.load(f)

    # Add nodes (blocked/high-risk checkpoints excluded up front)
    for node in topo['nodes']:
        if node['id'] not in blocked_checkpoints:
            G.add_node(node['id'], **node)

    config_path = os.path.join(os.path.dirname(topology_path), "multimodal_config.json")
    with open(config_path, encoding="utf-8") as f:
        modal_config = json.load(f)

    cargo_path = os.path.join(os.path.dirname(topology_path), "cargo_types.json")
    with open(cargo_path, encoding="utf-8") as f:
        cargo_types = {c["id"]: c for c in json.load(f)["types"]}

    # Add edges with cost weights (hard cargo constraints applied here,
    # before search, by simply never adding the disallowed edge)
    for edge in topo['edges']:
        if edge['from'] in G.nodes and edge['to'] in G.nodes:
            mode = edge['mode']
            mode_cfg = modal_config['modes'].get(mode, {})

            # Check cargo constraints
            max_weight = mode_cfg.get('max_cargo_weight_tons', 999)
            if cargo_weight_tons > max_weight:
                continue  # Skip this edge — cargo too heavy

            # Check hazmat
            if cargo_type == 'chemicals' and mode == 'rail':
                continue  # Hazmat restriction

            # Check cold chain
            cargo_cfg = cargo_types.get(cargo_type, {})
            if cargo_cfg.get('requires_cold_chain') and mode == 'sea':
                continue  # No cold chain on sea for this config

            # Weight = time + cost component + mode transfer penalty
            cost_km = mode_cfg.get('cost_per_km_inr', 35)
            time_weight = edge['base_time_min']
            cost_weight = edge['distance_km'] * cost_km / 1000  # Normalized

            # Additional risk weight could be dynamically calculated and updated here
            # We initialize risk_weight same as time_weight initially

            G.add_edge(
                edge['from'], edge['to'],
                mode=mode,
                distance_km=edge['distance_km'],
                base_time_min=edge['base_time_min'],
                cost_inr=edge['distance_km'] * cost_km,
                co2_kg=edge['distance_km'] * mode_cfg.get('co2_per_km_kg', 0.27),
                weight=time_weight + cost_weight,
                risk_weight=time_weight  # Placeholder for safety priority
            )

    return G


# Backward-compatible alias — this function was previously named
# `build_multimodal_graph`; kept so nothing else that may import it by
# the old name breaks.
build_multimodal_graph = build_constrained_graph


def get_edge_weight(G: nx.Graph, u: str, v: str, weight_attr: str) -> float:
    """Reads the g(n)-contribution of a single edge for the given objective."""
    return G[u][v][weight_attr]


def find_route_astar(G: nx.Graph, origin: str, destination: str,
                      weight_attr: str, priority: str,
                      node_coords: dict) -> Optional[list]:
    """
    Runs A* (f = g + h) once, returning the single optimal path from
    `origin` to `destination`, or None if no path exists.

    g(n) is accumulated via `weight_attr` (base_time_min / cost_inr /
    risk_weight, per `priority`) exactly as Dijkstra would accumulate it.
    h(n) comes from `astar_heuristic()`.
    """
    evaluations = 0

    def heuristic(u, v):
        nonlocal evaluations
        evaluations += 1
        return astar_heuristic(u, v, node_coords, priority)

    try:
        path = nx.astar_path(G, origin, destination, heuristic=heuristic, weight=weight_attr)
        logger.debug(
            "A* %s->%s priority=%s: found path len=%d, heuristic evaluations=%d",
            origin, destination, priority, len(path), evaluations,
        )
        return path
    except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
        logger.debug("A* %s->%s priority=%s: no path (%s)", origin, destination, priority, exc)
        return None


def _hidden_view(G: nx.Graph, nodes_to_hide: set, edges_to_hide: set) -> nx.Graph:
    """Read-only view of G with the given nodes/edges filtered out — used
    to force each A* re-run in find_k_routes_astar() onto a genuinely
    different path, without mutating (and having to painstakingly
    restore) the shared graph object."""
    def filter_node(n):
        return n not in nodes_to_hide

    def filter_edge(u, v):
        return (u, v) not in edges_to_hide and (v, u) not in edges_to_hide

    return nx.subgraph_view(G, filter_node=filter_node, filter_edge=filter_edge)


def find_k_routes_astar(G: nx.Graph, origin: str, destination: str,
                         weight_attr: str, priority: str,
                         node_coords: dict, k: int = 20) -> list:
    """
    Up to `k` ranked, loopless alternative routes, using A* as the
    pathfinding subroutine (Yen's-algorithm bookkeeping around it).

    IMPORTANT: A* itself only ever returns one optimal path per call —
    it is not a k-shortest-path algorithm. To still expose several
    ranked alternatives (as the dashboard requires), this function
    re-runs A* on progressively constrained views of the same graph,
    exactly the way NetworkX's own `shortest_simple_paths` re-runs
    Dijkstra internally. A* remains the algorithm doing every bit of
    the actual pathfinding; this loop only decides *which* source/graph
    to hand it next.
    """
    first = find_route_astar(G, origin, destination, weight_attr, priority, node_coords)
    if first is None:
        return []

    accepted = [first]
    candidates: list = []  # list of (cost, path)

    def path_cost(path):
        return sum(get_edge_weight(G, path[i], path[i + 1], weight_attr) for i in range(len(path) - 1))

    for _ in range(1, k):
        prev_path = accepted[-1]
        for i in range(len(prev_path) - 1):
            spur_node = prev_path[i]
            root_path = prev_path[: i + 1]

            edges_to_hide = {
                (path[i], path[i + 1])
                for path in accepted
                if len(path) > i and path[: i + 1] == root_path
            }
            nodes_to_hide = set(root_path[:-1])  # every root node except the spur node itself

            view = _hidden_view(G, nodes_to_hide, edges_to_hide)
            spur_path = find_route_astar(view, spur_node, destination, weight_attr, priority, node_coords)

            if spur_path:
                total_path = root_path[:-1] + spur_path
                if total_path not in accepted and all(total_path != p for _, p in candidates):
                    candidates.append((path_cost(total_path), total_path))

        if not candidates:
            break
        candidates.sort(key=lambda c: c[0])
        accepted.append(candidates.pop(0)[1])

    return accepted


def find_optimal_routes(origin: str, destination: str,
                         blocked: list = None,
                         priority: str = "speed",
                         top_k: int = 5,
                         cargo_type: str = "fmcg",
                         cargo_weight_tons: float = 1.0) -> list[dict]:
    if blocked is None:
        blocked = []

    if blocked:
        logger.info(
            "REROUTE requested: origin=%s destination=%s avoiding=%s priority=%s "
            "-> rebuilding graph and re-running A*",
            origin, destination, blocked, priority,
        )

    topology_path = os.path.join(os.path.dirname(__file__), "..", "data", "graph_topology.json")
    G = build_constrained_graph(topology_path, blocked, cargo_weight_tons, cargo_type)

    weight_attr = WEIGHT_ATTR_BY_PRIORITY.get(priority, "base_time_min")
    node_coords = {n: {"lat": data.get("lat"), "lon": data.get("lon")} for n, data in G.nodes(data=True)}

    logger.info(
        "A* route search: origin=%s destination=%s priority=%s weight_attr=%s "
        "heuristic=%s",
        origin, destination, priority, weight_attr,
        "haversine/%.0fkm_h" % MAX_PLAUSIBLE_SPEED_KMH if priority == "speed"
        else "haversine*min_INR_per_km" if priority == "cost"
        else "h=0 (Dijkstra-equivalent; risk is not a spatial quantity)",
    )

    # Get up to 20 ranked alternative paths, A* subroutine under the hood
    paths = find_k_routes_astar(G, origin, destination, weight_attr, priority, node_coords, k=20)
    if not paths:
        logger.info("A* route search: no path found from %s to %s", origin, destination)
        return []  # Edge case: no path

    routes = []
    config_path = os.path.join(os.path.dirname(topology_path), "multimodal_config.json")
    with open(config_path, encoding="utf-8") as f:
        modal_config = json.load(f)

    for i, path in enumerate(paths):
        edges = [G[path[j]][path[j+1]] for j in range(len(path)-1)]
        modes_used = list(dict.fromkeys(e['mode'] for e in edges))  # Ordered unique modes

        # Count mode transfers
        transfers = sum(1 for j in range(len(edges)-1) if edges[j]['mode'] != edges[j+1]['mode'])

        # Transfer penalties
        transfer_cost = 0
        transfer_time = 0

        for j in range(len(edges)-1):
            if edges[j]['mode'] != edges[j+1]['mode']:
                key = f"{edges[j]['mode']}_to_{edges[j+1]['mode']}"
                penalty = modal_config['transfer_penalties'].get(key, {})
                transfer_time += penalty.get('time_min', 60)
                transfer_cost += penalty.get('cost_inr', 2000)

        total_time = sum(e['base_time_min'] for e in edges) + transfer_time
        total_cost = sum(e['cost_inr'] for e in edges) + transfer_cost
        total_distance = sum(e['distance_km'] for e in edges)
        total_co2 = sum(e['co2_kg'] for e in edges)

        # --- DYNAMIC COST CALCULATION ---
        # 1. Cargo-type surcharge (+5% to +25%)
        cargo_surcharge = 0.0
        cargo_driver = ""
        if cargo_type in ['chemicals', 'automotive', 'pharmaceuticals']:
            cargo_surcharge = random.uniform(0.05, 0.25)
            cargo_driver = "Cargo Premium"

        # 2. Route distance tier volatility
        volatility = 0.15
        if total_distance < 200:
            volatility = 0.20
            distance_driver = "Short-Haul Volatility"
        elif total_distance > 500:
            volatility = 0.08
            distance_driver = "Long-Haul Contract"
        else:
            distance_driver = "Spot Rate"

        market_surge = random.uniform(-volatility, volatility)

        # 3. Time-of-day / Rush Hour
        hour = datetime.now().hour
        rush_hour_surge = 0.0
        if 8 <= hour <= 10 or 17 <= hour <= 19:
            rush_hour_surge = random.uniform(0.08, 0.15)
            time_driver = "Rush Hour"
        else:
            time_driver = ""

        # Determine main driver
        total_surge_pct = cargo_surcharge + market_surge + rush_hour_surge

        if cargo_surcharge > abs(market_surge) and cargo_surcharge > rush_hour_surge:
            driver = cargo_driver
        elif rush_hour_surge > abs(market_surge):
            driver = time_driver
        else:
            driver = distance_driver

        if total_surge_pct > 0.10:
            label = "High Demand Surge"
        elif total_surge_pct < -0.05:
            label = "Favorable Market Rate"
        else:
            label = "Standard Spot Rate"

        total_cost = total_cost * (1 + total_surge_pct)

        market_trend = {
            "label": label,
            "delta_pct": round(total_surge_pct * 100, 1),
            "driver": driver
        }

        routes.append({
            "route_id": f"ROUTE_{chr(65+i)}",
            "path": path,
            "modes": modes_used,
            "is_multimodal": len(modes_used) > 1,
            "transfers": transfers,
            "total_time_min": total_time,
            "total_cost_inr": round(total_cost, 2),
            "total_distance_km": round(total_distance, 2),
            "total_co2_kg": round(total_co2, 2),
            "market_trend": market_trend,
            "checkpoints": [n for n in path if n.startswith('CP') or n.startswith('RN') or n.startswith('PT') or n.startswith('AP')]
        })

    # Sort the evaluated routes based on priority
    if priority == "speed":
        routes.sort(key=lambda x: x["total_time_min"])
    elif priority == "cost":
        routes.sort(key=lambda x: x["total_cost_inr"])
    else:
        # Default safety/time combo
        routes.sort(key=lambda x: x["total_time_min"] + (x["transfers"] * 30))

    # Re-assign route IDs so the best is always ROUTE_A
    for i, r in enumerate(routes):
        r["route_id"] = f"ROUTE_{chr(65+i)}"

    # Filter out illogical routes that are > 2.5x worse than the best route
    if not routes:
        return []

    best_time = routes[0]["total_time_min"]
    best_cost = routes[0]["total_cost_inr"]

    valid_routes = []
    for r in routes:
        # If a route is more than 2.5x the time AND 2.5x the cost, it's illogical
        if r["total_time_min"] <= best_time * 2.5 or r["total_cost_inr"] <= best_cost * 2.5:
            valid_routes.append(r)

    result = valid_routes[:top_k]
    logger.info(
        "A* route search complete: origin=%s destination=%s priority=%s -> %d route(s), "
        "best=%s (%.1f min, Rs.%.2f) rerouted=%s",
        origin, destination, priority, len(result),
        result[0]["route_id"] if result else None,
        result[0]["total_time_min"] if result else -1,
        result[0]["total_cost_inr"] if result else -1,
        bool(blocked),
    )
    return result
