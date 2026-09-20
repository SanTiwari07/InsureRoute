"""
test_graph_router.py — Tests for the A*-based route planner.

Run with:
    python -m unittest test_graph_router -v
(from the backend/ directory; no extra test framework required — this
project has no pytest dependency installed, so these use the standard
library's unittest.)
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

import networkx as nx

from core.graph_router import (
    astar_heuristic,
    build_constrained_graph,
    find_k_routes_astar,
    find_optimal_routes,
    find_route_astar,
    haversine_distance,
)

TOPOLOGY_PATH = os.path.join(os.path.dirname(__file__), "data", "graph_topology.json")


def _node_coords(G):
    return {n: {"lat": d.get("lat"), "lon": d.get("lon")} for n, d in G.nodes(data=True)}


class HaversineTests(unittest.TestCase):
    def test_zero_distance_to_self(self):
        self.assertAlmostEqual(haversine_distance(18.52, 73.85, 18.52, 73.85), 0.0, places=6)

    def test_symmetric(self):
        d1 = haversine_distance(18.5204, 73.8567, 19.0760, 72.8777)
        d2 = haversine_distance(19.0760, 72.8777, 18.5204, 73.8567)
        self.assertAlmostEqual(d1, d2, places=6)

    def test_pune_to_mumbai_is_a_plausible_straight_line_distance(self):
        # Real-world straight-line Pune<->Mumbai distance is ~120-150 km.
        # This is a sanity bound, not a precision assertion.
        km = haversine_distance(18.5204, 73.8567, 19.0760, 72.8777)
        self.assertGreater(km, 90)
        self.assertLess(km, 170)


class HeuristicAdmissibilityTests(unittest.TestCase):
    """The core optimality claim of A* rests on h(n) never overestimating
    the true remaining cost. These tests verify that empirically against
    every reachable node in the actual topology, for every priority whose
    heuristic is not simply h(n)=0."""

    @classmethod
    def setUpClass(cls):
        cls.G = build_constrained_graph(TOPOLOGY_PATH)
        cls.coords = _node_coords(cls.G)

    def _assert_never_overestimates(self, weight_attr, priority):
        destination = "MUMBAI_DEST"
        checked = 0
        for node in self.G.nodes:
            if not nx.has_path(self.G, node, destination):
                continue
            true_cost = nx.dijkstra_path_length(self.G, node, destination, weight=weight_attr)
            h = astar_heuristic(node, destination, self.coords, priority)
            self.assertLessEqual(
                h, true_cost + 1e-9,
                msg=f"h({node})={h} overestimates true remaining {priority} cost {true_cost}",
            )
            checked += 1
        self.assertGreater(checked, 0, "sanity: the test actually exercised some nodes")

    def test_speed_heuristic_is_admissible(self):
        self._assert_never_overestimates("base_time_min", "speed")

    def test_cost_heuristic_is_admissible(self):
        self._assert_never_overestimates("cost_inr", "cost")

    def test_safety_heuristic_is_zero(self):
        # risk_weight is not a spatial quantity, so h(n) must be exactly 0
        # (A* reduces to Dijkstra) rather than a guessed geographic value.
        for node in list(self.G.nodes)[:5]:
            self.assertEqual(astar_heuristic(node, "MUMBAI_DEST", self.coords, "safety"), 0.0)


class AStarBasicCorrectnessTests(unittest.TestCase):
    """A. Basic A* correctness: known optimal route + total weight, from
    the actual (hand-traced) graph_topology.json."""

    @classmethod
    def setUpClass(cls):
        cls.G = build_constrained_graph(TOPOLOGY_PATH, cargo_type="fmcg", cargo_weight_tons=1.0)
        cls.coords = _node_coords(cls.G)

    def test_fastest_route_pune_to_mumbai_is_the_air_leg(self):
        # PUNE_DEPOT->AP01 (20) ->PT01 (45, air) ->MUMBAI_DEST (25) = 90 min,
        # which beats every all-road (>=213 min) and rail (233 min) option
        # in this topology.
        path = find_route_astar(self.G, "PUNE_DEPOT", "MUMBAI_DEST", "base_time_min", "speed", self.coords)
        self.assertEqual(path, ["PUNE_DEPOT", "AP01", "PT01", "MUMBAI_DEST"])
        total = sum(self.G[path[i]][path[i + 1]]["base_time_min"] for i in range(len(path) - 1))
        self.assertEqual(total, 90)

    def test_astar_matches_plain_dijkstra_cost(self):
        # A* with an admissible heuristic must return a path whose true
        # cost equals Dijkstra's optimum, for every objective.
        for weight_attr, priority in [("base_time_min", "speed"), ("cost_inr", "cost"), ("risk_weight", "safety")]:
            astar_path = find_route_astar(self.G, "PUNE_DEPOT", "MUMBAI_DEST", weight_attr, priority, self.coords)
            astar_cost = sum(self.G[astar_path[i]][astar_path[i + 1]][weight_attr] for i in range(len(astar_path) - 1))
            dijkstra_cost = nx.dijkstra_path_length(self.G, "PUNE_DEPOT", "MUMBAI_DEST", weight=weight_attr)
            self.assertAlmostEqual(astar_cost, dijkstra_cost, places=6, msg=f"mismatch for priority={priority}")


class ObjectiveTests(unittest.TestCase):
    """B. Different objectives (speed / cost / safety) must use their own
    edge weight and can legitimately produce different optimal routes."""

    @classmethod
    def setUpClass(cls):
        cls.G = build_constrained_graph(TOPOLOGY_PATH, cargo_type="fmcg", cargo_weight_tons=1.0)
        cls.coords = _node_coords(cls.G)

    def test_speed_and_cost_optimize_different_routes(self):
        fastest = find_route_astar(self.G, "PUNE_DEPOT", "MUMBAI_DEST", "base_time_min", "speed", self.coords)
        cheapest = find_route_astar(self.G, "PUNE_DEPOT", "MUMBAI_DEST", "cost_inr", "cost", self.coords)
        # Fastest = the pricey air leg; cheapest = the slow rail leg.
        self.assertIn("AP01", fastest)
        self.assertIn("RN01", cheapest)
        self.assertNotEqual(fastest, cheapest)

    def test_all_three_priorities_return_a_valid_path(self):
        for priority, weight_attr in [("speed", "base_time_min"), ("cost", "cost_inr"), ("safety", "risk_weight")]:
            path = find_route_astar(self.G, "PUNE_DEPOT", "MUMBAI_DEST", weight_attr, priority, self.coords)
            self.assertIsNotNone(path)
            self.assertEqual(path[0], "PUNE_DEPOT")
            self.assertEqual(path[-1], "MUMBAI_DEST")


class BlockedCheckpointReroutingTests(unittest.TestCase):
    """C. Blocking a checkpoint (simulating live ML/weather risk) must
    remove it from the graph BEFORE search, forcing a genuinely
    different A* result — this is the "reroute" behaviour end to end."""

    def test_blocking_the_air_hub_forces_a_different_fastest_route(self):
        G_open = build_constrained_graph(TOPOLOGY_PATH)
        coords = _node_coords(G_open)
        original = find_route_astar(G_open, "PUNE_DEPOT", "MUMBAI_DEST", "base_time_min", "speed", coords)
        self.assertIn("AP01", original)

        G_blocked = build_constrained_graph(TOPOLOGY_PATH, blocked_checkpoints=["AP01"])
        coords_blocked = _node_coords(G_blocked)
        rerouted = find_route_astar(G_blocked, "PUNE_DEPOT", "MUMBAI_DEST", "base_time_min", "speed", coords_blocked)

        self.assertIsNotNone(rerouted)
        self.assertNotIn("AP01", rerouted)
        self.assertNotEqual(original, rerouted)

    def test_find_optimal_routes_end_to_end_reroute(self):
        # Full public entry point used by the API / Gemini agent's
        # trigger_reroute tool. Note: the air leg has the lowest raw
        # base_time_min (90 min) but incurs large road<->air transfer
        # penalties (165 min) applied during ranking, so it is not
        # necessarily route A once fully costed — it should still appear
        # somewhere in the ranked alternatives before being blocked.
        baseline = find_optimal_routes("PUNE_DEPOT", "MUMBAI_DEST", priority="speed", top_k=20)
        self.assertTrue(baseline)
        self.assertTrue(
            any("AP01" in r["path"] for r in baseline),
            "the air route should appear as one of the ranked alternatives",
        )

        rerouted = find_optimal_routes("PUNE_DEPOT", "MUMBAI_DEST", blocked=["AP01"], priority="speed")
        self.assertTrue(rerouted)
        for route in rerouted:
            self.assertNotIn("AP01", route["path"])


class CargoConstraintTests(unittest.TestCase):
    """D. Hard cargo constraints must be enforced by never adding the
    disallowed edge (i.e. before A* runs), not by filtering afterwards."""

    def test_hazmat_chemicals_cannot_use_the_rail_edge(self):
        G = build_constrained_graph(TOPOLOGY_PATH, cargo_type="chemicals")
        self.assertFalse(G.has_edge("RN01", "RN02"))

    def test_non_hazmat_cargo_can_use_the_rail_edge(self):
        G = build_constrained_graph(TOPOLOGY_PATH, cargo_type="fmcg")
        self.assertTrue(G.has_edge("RN01", "RN02"))

    def test_overweight_cargo_cannot_use_the_air_edge(self):
        # air mode's max_cargo_weight_tons is 5 in multimodal_config.json
        G = build_constrained_graph(TOPOLOGY_PATH, cargo_type="fmcg", cargo_weight_tons=10.0)
        self.assertFalse(G.has_edge("AP01", "PT01"))

    def test_route_search_never_uses_a_constraint_violating_edge(self):
        G = build_constrained_graph(TOPOLOGY_PATH, cargo_type="chemicals")
        coords = _node_coords(G)
        path = find_route_astar(G, "PUNE_DEPOT", "MUMBAI_DEST", "cost_inr", "cost", coords)
        edges_used = set(zip(path, path[1:])) | set(zip(path[1:], path))
        self.assertNotIn(("RN01", "RN02"), edges_used)


class AlternativeRoutesTests(unittest.TestCase):
    """Alternatives are still available (frontend renders several ranked
    options), now produced by re-running A* Yen's-style instead of
    nx.shortest_simple_paths."""

    def test_multiple_ranked_alternatives_are_returned(self):
        G = build_constrained_graph(TOPOLOGY_PATH)
        coords = _node_coords(G)
        paths = find_k_routes_astar(G, "PUNE_DEPOT", "MUMBAI_DEST", "base_time_min", "speed", coords, k=5)
        self.assertGreater(len(paths), 1)
        # Strictly non-decreasing total cost, i.e. genuinely ranked.
        costs = [sum(G[p[i]][p[i + 1]]["base_time_min"] for i in range(len(p) - 1)) for p in paths]
        self.assertEqual(costs, sorted(costs))
        # All loopless and all distinct.
        for p in paths:
            self.assertEqual(len(p), len(set(p)))
        self.assertEqual(len(paths), len(set(tuple(p) for p in paths)))


class RegressionTests(unittest.TestCase):
    """F. The public API surface (find_optimal_routes) must keep its
    existing signature and response schema."""

    def test_response_schema_unchanged(self):
        routes = find_optimal_routes("PUNE_DEPOT", "MUMBAI_DEST", priority="speed", top_k=5, cargo_type="fmcg", cargo_weight_tons=1.0)
        self.assertTrue(routes)
        expected_keys = {
            "route_id", "path", "modes", "is_multimodal", "transfers",
            "total_time_min", "total_cost_inr", "total_distance_km",
            "total_co2_kg", "market_trend", "checkpoints",
        }
        for route in routes:
            self.assertEqual(expected_keys, set(route.keys()))
            self.assertEqual({"label", "delta_pct", "driver"}, set(route["market_trend"].keys()))

        self.assertEqual(routes[0]["route_id"], "ROUTE_A")
        self.assertLessEqual(len(routes), 5)

    def test_routes_are_sorted_by_time_for_speed_priority(self):
        routes = find_optimal_routes("PUNE_DEPOT", "MUMBAI_DEST", priority="speed")
        times = [r["total_time_min"] for r in routes]
        self.assertEqual(times, sorted(times))

    def test_unreachable_destination_returns_empty_list_not_an_exception(self):
        all_intermediate = ["CP01", "CP02", "CP03", "CP04", "CP05", "CP06", "CP07", "CP08",
                             "RN01", "RN02", "AP01", "PT01"]
        routes = find_optimal_routes("PUNE_DEPOT", "MUMBAI_DEST", blocked=all_intermediate)
        self.assertEqual(routes, [])

    def test_default_top_k_and_priority_still_work(self):
        # Mirrors how main.py's create_shipment() calls this today.
        routes = find_optimal_routes("PUNE_DEPOT", "MUMBAI_DEST")
        self.assertTrue(routes)


CARGO_VARIANTS = [("fmcg", 1.0), ("chemicals", 1.0), ("perishables", 1.0), ("fmcg", 10.0), ("fmcg", 40.0)]
NON_ZERO_HEURISTICS = [("speed", "base_time_min"), ("cost", "cost_inr")]


class HeuristicPremiseTests(unittest.TestCase):
    """The admissibility/consistency proof rests on ONE per-edge premise:
        weight(u, v) >= c * great_circle_km(u, v)
    which is a property of the topology DATA, not of physics (the declared
    distance_km is below the great-circle distance on several edges). If
    graph_topology.json / multimodal_config.json are edited so this stops
    holding, these tests fail instead of A* silently returning
    non-optimal routes."""

    def test_every_edge_satisfies_the_premise_the_heuristic_relies_on(self):
        for cargo, wt in CARGO_VARIANTS:
            G = build_constrained_graph(TOPOLOGY_PATH, [], wt, cargo)
            coords = _node_coords(G)
            for priority, attr in NON_ZERO_HEURISTICS:
                for u, v, data in G.edges(data=True):
                    # h(u -> dest=v) == c * great_circle(u, v)
                    self.assertLessEqual(
                        astar_heuristic(u, v, coords, priority), data[attr] + 1e-9,
                        msg=f"premise violated on edge {u}-{v} for priority={priority}",
                    )

    def test_admissible_and_consistent_for_every_destination_and_cargo_variant(self):
        for cargo, wt in CARGO_VARIANTS:
            G = build_constrained_graph(TOPOLOGY_PATH, [], wt, cargo)
            coords = _node_coords(G)
            for priority, attr in NON_ZERO_HEURISTICS:
                for dest in G.nodes:
                    for n in G.nodes:
                        if nx.has_path(G, n, dest):
                            true_cost = nx.dijkstra_path_length(G, n, dest, weight=attr)
                            self.assertLessEqual(astar_heuristic(n, dest, coords, priority), true_cost + 1e-9)
                    for u, v, data in G.edges(data=True):
                        for x, y in ((u, v), (v, u)):
                            # consistency: h(x) <= w(x, y) + h(y)
                            self.assertLessEqual(
                                astar_heuristic(x, dest, coords, priority) - astar_heuristic(y, dest, coords, priority),
                                data[attr] + 1e-9,
                            )

    def test_unknown_node_or_safety_priority_falls_back_to_zero(self):
        self.assertEqual(astar_heuristic("NOPE", "MUMBAI_DEST", {}, "speed"), 0.0)
        self.assertEqual(astar_heuristic("CP01", "MUMBAI_DEST", {"CP01": {"lat": 1, "lon": 1}}, "cost"), 0.0)


def _count_expansions_astar(G, source, target, attr, heuristic):
    """Nodes whose outgoing edges get relaxed == nodes A* expanded."""
    expanded = set()

    def weight(u, v, d):
        expanded.add(u)
        return d[attr]

    nx.astar_path(G, source, target, heuristic=heuristic, weight=weight)
    return len(expanded)


def _count_expansions_dijkstra(G, source, target, attr):
    expanded = set()

    def weight(u, v, d):
        expanded.add(u)
        return d[attr]

    nx.dijkstra_path(G, source, target, weight=weight)
    return len(expanded)


class ExpansionComparisonTests(unittest.TestCase):
    """A* with a consistent heuristic must never expand more nodes than
    Dijkstra IN TOTAL. We deliberately do not assert 'strictly fewer':
    on this 14-node graph the saving is marginal (and zero for the
    Pune->Mumbai request) — see AIML_PROJECT_REPORT.md section 6.9."""

    def test_astar_total_expansions_do_not_exceed_dijkstra(self):
        G = build_constrained_graph(TOPOLOGY_PATH)
        coords = _node_coords(G)
        for priority, attr in NON_ZERO_HEURISTICS:
            total_astar = total_dijkstra = 0
            for s in G.nodes:
                for t in G.nodes:
                    if s == t or not nx.has_path(G, s, t):
                        continue
                    h = lambda u, v, p=priority: astar_heuristic(u, v, coords, p)
                    total_astar += _count_expansions_astar(G, s, t, attr, h)
                    total_dijkstra += _count_expansions_dijkstra(G, s, t, attr)
            self.assertLessEqual(total_astar, total_dijkstra, msg=f"priority={priority}")


class EquivalenceWithPreviousAlgorithmTests(unittest.TestCase):
    """The A*-based alternatives must reproduce what the previous
    nx.shortest_simple_paths implementation returned (same paths, same
    cost ranking) — the migration must not change which routes exist."""

    def test_same_candidate_paths_and_cost_ranking_as_shortest_simple_paths(self):
        import itertools
        G = build_constrained_graph(TOPOLOGY_PATH)
        coords = _node_coords(G)
        for priority, attr in [("speed", "base_time_min"), ("cost", "cost_inr"), ("safety", "risk_weight")]:
            new = find_k_routes_astar(G, "PUNE_DEPOT", "MUMBAI_DEST", attr, priority, coords, k=20)
            old = list(itertools.islice(nx.shortest_simple_paths(G, "PUNE_DEPOT", "MUMBAI_DEST", weight=attr), 20))
            cost = lambda p: sum(G[p[i]][p[i + 1]][attr] for i in range(len(p) - 1))
            self.assertEqual({tuple(p) for p in new}, {tuple(p) for p in old}, msg=priority)
            self.assertEqual([round(cost(p), 6) for p in new], [round(cost(p), 6) for p in old], msg=priority)


class LivePipelineWiringTests(unittest.TestCase):
    """Proves the public entry point actually executes nx.astar_path — i.e.
    A* is not dead code — and that the old Yen/Dijkstra call is gone."""

    def _run_with_spies(self, **kwargs):
        from unittest import mock
        with mock.patch.object(nx, "astar_path", wraps=nx.astar_path) as astar_spy, \
             mock.patch.object(nx, "shortest_simple_paths",
                               side_effect=AssertionError("legacy shortest_simple_paths must not be called")):
            routes = find_optimal_routes("PUNE_DEPOT", "MUMBAI_DEST", **kwargs)
        return routes, astar_spy

    def test_initial_search_runs_astar(self):
        routes, spy = self._run_with_spies(priority="speed")
        self.assertTrue(routes)
        self.assertGreater(spy.call_count, 0)

    def test_reroute_runs_astar_again_on_the_reduced_graph(self):
        routes, spy = self._run_with_spies(priority="speed", blocked=["CP03"])
        self.assertTrue(routes)
        self.assertGreater(spy.call_count, 0)
        for r in routes:
            self.assertNotIn("CP03", r["path"])

    def test_every_objective_goes_through_astar(self):
        for priority in ("speed", "cost", "safety"):
            _, spy = self._run_with_spies(priority=priority)
            self.assertGreater(spy.call_count, 0, msg=priority)


class ApiHandlerRegressionTests(unittest.TestCase):
    """Existing API surface still works end to end (handlers imported
    directly; the HTTP layer test is skipped if httpx is unavailable)."""

    @classmethod
    def setUpClass(cls):
        try:
            import main  # noqa: WPS433 — heavy import, so done lazily
        except ImportError as exc:  # e.g. google-generativeai not installed
            raise unittest.SkipTest(f"backend app not importable here: {exc}")
        cls.main = main

    def test_create_shipment_handler_contract(self):
        from models.schemas import ShipmentCreate
        resp = self.main.create_shipment(ShipmentCreate(
            origin="PUNE_DEPOT", destination="MUMBAI_DEST", cargo_type="electronics",
            cargo_value_inr=500000, cargo_weight_tons=2.0, priority="speed"))
        self.assertEqual({"shipment_id", "options", "cargo_value_inr", "cargo_type"}, set(resp.keys()))
        self.assertTrue(resp["options"])
        self.assertEqual(resp["options"][0]["route_id"], "ROUTE_A")

    def test_gemini_trigger_reroute_tool_avoids_blocked_checkpoint(self):
        route = self.main.trigger_reroute_tool("PUNE_DEPOT", "MUMBAI_DEST", ["CP03"], "electronics", "speed")
        self.assertNotIn("error", route)
        self.assertNotIn("CP03", route["path"])

    def test_http_create_shipment_endpoint(self):
        try:
            from fastapi.testclient import TestClient
        except Exception as exc:
            self.skipTest(f"TestClient unavailable (httpx missing?): {exc}")
        client = TestClient(self.main.app)  # no `with`: startup tasks are not run
        r = client.post("/api/v1/shipments/create", json={
            "origin": "PUNE_DEPOT", "destination": "MUMBAI_DEST", "cargo_type": "fmcg",
            "cargo_value_inr": 100000, "cargo_weight_tons": 1.0, "priority": "cost"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        for opt in body["options"]:
            self.assertTrue({"path", "modes", "total_time_min", "total_cost_inr", "market_trend"} <= set(opt))


if __name__ == "__main__":
    unittest.main()
