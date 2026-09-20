# InsureRoute — AI/ML Project Report

**Smart Supply Chain Disruption Detection & Dynamic Insurance Pricing**
Prepared for: AIML Subject Activity/Project Submission
Repository: [InsureRoute](https://github.com/kanishkasalgude/InsureRoute)

---

## 1. Abstract

InsureRoute is a full-stack logistics-intelligence platform that predicts transit risk on a cargo route, re-plans the route when risk crosses a threshold, prices an insurance premium against that risk in real time, and explains all of it to a human operator in natural language. It is deliberately built as a **hybrid AI system** — it does not rely on a single model, but combines four distinct AI/ML paradigms that are normally taught as separate units in an AIML course:

| # | Paradigm | Where it lives | Course topic it demonstrates |
|---|---|---|---|
| 1 | Supervised Learning (regression, ensemble) | [`backend/core/ml_engine.py`](backend/core/ml_engine.py) | Random Forests, regression, feature engineering |
| 2 | Unsupervised Learning (anomaly detection) | [`backend/legacy/model.py`](backend/legacy/model.py) | Isolation Forest, outlier detection |
| 3 | Classical AI Search | [`backend/core/graph_router.py`](backend/core/graph_router.py) | A* search, admissible heuristics, state-space re-planning |
| 4 | Generative / Agentic AI | [`backend/core/gemini_agent.py`](backend/core/gemini_agent.py) | LLMs, prompt engineering, tool-calling agents |

A fifth component — the [actuarial pricing engine](backend/core/insurance_engine.py) — is **deliberately rule-based, not learned**, which is itself a useful discussion point about when *not* to use ML (see §8).

This document walks through **what was built, which algorithm was chosen for each sub-problem, and why**, so it can be presented as a project artifact for an AIML activity.

---

## 2. Problem Statement

Cargo insurance today is static: a shipment is quoted once, at a flat rate, regardless of the actual conditions it will pass through. Meanwhile, logistics disruptions (weather, accidents, strikes) are detected reactively — after delay has already occurred. The project asks: **can a route's risk be predicted continuously from live signals, can the system react by re-routing automatically, and can the insurance premium be repriced in real time to reflect the actual risk taken?**

This is naturally decomposed into four AI sub-problems:

1. **Prediction** — given weather, time, traffic and checkpoint-type signals, estimate a risk score for a location/segment. *(Supervised learning)*
2. **Detection** — given historical transit records with no risk labels available, flag which ones look abnormal. *(Unsupervised learning)*
3. **Planning** — given a graph of transport nodes and a risk-weighted cost function, find the best path using an informed search that exploits the nodes' real GPS coordinates, and re-find it when conditions change. *(A\* search/planning)*
4. **Explanation & decisioning** — given the numeric outputs of (1)–(3), produce a human-readable recommendation and decide whether to act (reroute / escalate coverage). *(Generative AI / agentic reasoning)*

---

## 3. System Architecture (Data Flow)

```
User Action (React Dashboard)
        │
        ▼
[FastAPI Gateway]  ──► NetworkX Graph Engine + A* Search (Component 3: Search)
        │             ──► OpenWeatherMap live telemetry
        │             ──► RandomForestRegressor      (Component 1: Supervised ML)
        │             ──► Insurance Rule Engine       (Component 5: Rule-based)
        │             ──► Gemini 2.5 Flash Agent      (Component 4: Agentic AI)
        ▼
[Frontend] renders KPIs, animated route, AI explanation, live quote
```

Each engine is an independently testable Python module (`backend/core/*.py`), so it maps cleanly onto separate "experiments" for a report: you can show the ML model's feature importances in isolation, show the routing engine re-planning in isolation, and show the LLM agent's tool-call trace in isolation.

---

## 4. Component 1 — Supervised Learning: Risk Scoring

**File:** [`backend/core/ml_engine.py`](backend/core/ml_engine.py)

### 4.1 Problem framing
Risk is modelled as **regression**, not classification: the model outputs a continuous score in `[0, 1]` rather than a discrete label (`safe` / `risky`). This was chosen deliberately because:
- Downstream consumers (the pricing engine, the routing engine's edge weights) need a continuous multiplier, not a bucket — an insurance premium computed on a binary flag would jump discontinuously between shipments that are almost identical in risk.
- A regression target also lets the report demonstrate **regression evaluation** (MAE/RMSE, feature importance) rather than only classification metrics — deliberately covering a different part of the AIML syllabus than the anomaly-detection component (§5).

### 4.2 Why Random Forest, and not a single Decision Tree or a neural net
| Candidate | Verdict | Reason |
|---|---|---|
| Single Decision Tree | Rejected | High variance — overfits the exact synthetic thresholds used to label training data |
| **Random Forest Regressor** | **Chosen** | Averages many de-correlated trees (bagging), which reduces variance without needing extra data; handles the mix of numeric (`rainfall_mm`, `wind_speed_kmh`) and encoded-categorical (`checkpoint_type_enc`, `road_condition_enc`) features natively, with no scaling step required |
| Linear/Logistic Regression | Rejected | The true risk surface generated by the synthetic data (§4.5) is explicitly non-linear (interaction of monsoon season × rainfall × traffic), which a linear model cannot capture without manual feature crosses |
| Neural network (MLP) | Rejected for now | Needs far more data than the 30,000-row synthetic set to avoid overfitting, is harder to explain to a non-technical operator, and offers no accuracy benefit at this feature-count/data-size regime. Listed under Future Scope (§10) once real telemetry volume grows |

This is a good "trade-off table" to defend in a viva: it shows *why* an ensemble tree method is the right complexity level for tabular data of this size, rather than reflexively reaching for a deep model.

### 4.3 Hyperparameters (and why)
```python
RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
```
- `n_estimators=100` — enough trees for the ensemble average to stabilise (variance reduction plateaus well before 100 for this feature count); more trees past this point cost training time for negligible accuracy gain.
- `max_depth=12` — caps tree depth to control overfitting; with 10 input features, depth 12 is enough to model feature interactions (e.g. rainfall × traffic × time-of-day) without individual trees memorising noise in a 30k-row dataset.
- `random_state=42` — fixes the random seed so training is **reproducible** — a requirement for any experiment section in an academic report (re-running the notebook must give the same feature importances).
- `n_jobs=-1` — parallelises tree construction across CPU cores; a systems-level engineering choice, not a modelling one, included here to show awareness of training-time cost.

### 4.4 Feature set and why each feature was included
| Feature | Type | Why it's predictive |
|---|---|---|
| `hour_of_day`, `day_of_week` | Temporal | Captures rush-hour traffic and weekday/weekend delivery patterns |
| `weather_code`, `rainfall_mm` | Live weather | Direct physical cause of transit delay/damage |
| `visibility_km`, `wind_speed_kmh` | Live weather | Proxies for accident risk (fog, storms) beyond rainfall alone |
| `traffic_density` | Heuristic | Congestion is a leading cause of ETA slippage independent of weather |
| `checkpoint_type_enc` | Categorical (encoded) | Road/rail/port/airport checkpoints carry structurally different risk profiles |
| `road_condition_enc`, `incident_type_enc` | Categorical (encoded) | Direct signal used at training time; at inference time these default to `'good'`/`'none'` when unknown, so the model degrades gracefully rather than failing |

**Encoding choice:** `sklearn.preprocessing.LabelEncoder` (integer encoding) was used instead of one-hot encoding. This is defensible specifically *because* the downstream model is tree-based — trees split on a single encoded integer threshold at a time, so they don't suffer from the "false ordinality" problem that hurts linear models with label-encoded categoricals. One-hot would have been the correct choice if a linear model had been used instead (worth stating explicitly in a viva, since examiners often probe this).

**Handling unseen labels at inference (`safe_encode`)** — a small but important production-ML detail: if a checkpoint reports a `road_condition` value the encoder never saw during training, `LabelEncoder.transform` throws. The code catches this and falls back to a default encoding (index `0`) rather than crashing the whole request — a simple but real example of **graceful degradation** in an ML-serving pipeline.

### 4.5 Training data: synthetic data generation
**File:** [`backend/scripts/generate_synthetic_data.py`](backend/scripts/generate_synthetic_data.py)

Real historical Indian logistics telemetry (weather-linked delay records, per-checkpoint) is not publicly available at the granularity needed, so training data (30,000 rows) is procedurally generated using domain-informed statistical distributions rather than pure random noise:

| Signal | Distribution used | Domain justification |
|---|---|---|
| `rainfall_mm` | `Exponential(2) × monsoon_factor` | Rainfall events are naturally right-skewed (many dry/light days, occasional heavy downpour) — an exponential distribution matches this shape far better than a Gaussian |
| `wind_speed_kmh` | `Normal(20, 10) × monsoon_factor` | Wind speed clusters around a mean with roughly symmetric variation |
| `traffic_density` | `Beta(2, 5) × peak_hour_factor` | Beta distribution is bounded in `[0,1]`, matching how density is *defined* (a ratio), and its shape is naturally skewed toward moderate congestion with occasional heavy jams |
| `visibility_km` | Derived from rainfall + Gaussian noise | Encodes the physical dependency between rain and visibility rather than treating them as independent |
| `risk_score` (label) | Weighted linear combination of rainfall, traffic, delay, and incident flag | Defines the **ground truth** the Random Forest is trained to approximate |

A `monsoon_flag` multiplier (June–September) and a `peak_hour` multiplier (8–10 AM, 5–8 PM) are injected so the generated dataset has the same seasonal/diurnal structure a real Indian logistics dataset would show — this lets the trained model's feature importances be sanity-checked against real-world intuition (e.g., "does the model learn that monsoon months are riskier?").

> **Honesty note for the report:** because labels are synthetically generated from a *known* formula, the Random Forest is effectively re-learning an approximation of that formula — evaluation accuracy will look artificially strong. This is called out explicitly in §9 (Limitations) as a discussion point, since acknowledging it is exactly what an AIML examiner looks for.

---

## 5. Component 2 — Unsupervised Learning: Anomaly Detection

**File:** [`backend/legacy/model.py`](backend/legacy/model.py) + [`backend/legacy/preprocessing.py`](backend/legacy/preprocessing.py)

This module represents the project's *first-principles* approach to the same problem — before the team decided a supervised regression target was more useful for pricing — and is kept in the codebase specifically because it demonstrates a genuinely different ML paradigm worth including in a report.

### 5.1 Why unsupervised at all
In a real deployment, nobody hand-labels "this shipment was disrupted" for every historical row — disruption is rare, expensive to confirm, and often ambiguous. **Isolation Forest** was chosen because it needs **no labels at all**: it learns what "normal" transit looks like and flags whatever is structurally different.

### 5.2 Why Isolation Forest specifically (over other unsupervised options)
| Candidate | Verdict | Reason |
|---|---|---|
| **Isolation Forest** | **Chosen** | Isolates anomalies by how *few* random splits it takes to separate a point from the rest of the data — anomalies sit in sparse regions of feature space and get isolated in fewer splits. This makes it `O(n log n)`, so it scales to large fleets, and it needs no distance metric (unlike k-NN/DBSCAN), so it isn't distorted by mixed-scale features once they're standardised |
| K-Means clustering | Rejected | Requires choosing `k` in advance and assumes roughly spherical clusters — disruption events don't form a clean cluster, they're scattered outliers |
| DBSCAN | Rejected | Density-based methods are sensitive to the `eps` hyperparameter and degrade in higher-dimensional feature spaces; Isolation Forest is more robust here with 7 engineered features |
| Autoencoder reconstruction error | Rejected for now | Needs meaningfully more data to train a reliable encoder without overfitting; overkill for 7 tabular features. Reasonable next step once IoT telemetry volume increases |

### 5.3 Hyperparameters
```python
IsolationForest(n_estimators=200, contamination=0.08, random_state=42, n_jobs=-1)
```
- `contamination=0.08` — the *prior belief* about what fraction of transit events are disruptions (~8%), used to calibrate the internal decision threshold. This is a domain assumption, not a learned value — worth flagging in a report as a hyperparameter that should ideally be tuned against real incident-rate statistics.
- `n_estimators=200` — more trees than the Random Forest regressor (§4.3) because anomaly isolation is a randomized process per tree; averaging over more trees stabilises the anomaly score itself.
- Decision rule: `anomaly_score_model < -0.15 → disruption_predicted = 1`. The `-0.15` cutoff is a manually chosen operating point on the model's raw `decision_function` output — again, an explicit, inspectable rule layered on top of the model rather than a hidden black-box threshold, which is good practice to point out in a viva on explainability.

### 5.4 Feature engineering pipeline
**File:** [`backend/legacy/preprocessing.py`](backend/legacy/preprocessing.py)

Unlike the Random Forest pipeline (which uses raw + encoded features directly, §4.4), this pipeline **engineers time-aware and derived features**:
- `delay_ratio = actual_transit_hrs / scheduled_transit_hrs` — normalises delay across routes of different lengths, so a 30-minute delay on a 1-hour trip and a 30-minute delay on a 10-hour trip are correctly weighted differently.
- `rolling_delay_mean_6h`, `rolling_delay_std_6h` — a 6-observation rolling window over `delay_ratio`, giving the model short-term *trend* information (is this checkpoint's performance degrading right now?) rather than only an instantaneous snapshot.
- All features are passed through `StandardScaler` (zero mean, unit variance) **before** fitting.

**Why scaling is required here but not in §4:** Isolation Forest's split points are chosen uniformly at random across each feature's *observed range*, so a feature with a naturally larger numeric range (e.g. `cargo_value_inr` in the tens of thousands vs `monsoon_flag` in `{0,1}`) would dominate the isolation process purely due to scale, not actual signal strength. Standardising removes that artifact. The Random Forest in §4 doesn't need this because tree splits are threshold comparisons on one feature at a time and are scale-invariant. **This contrast (why RF needs no scaling but Isolation Forest does) is one of the strongest "explain the choice" points to raise verbally in an AIML viva.**

---

## 6. Component 3 — Classical AI: A* Search & Route Re-planning

**File:** [`backend/core/graph_router.py`](backend/core/graph_router.py)

This is not machine learning — it is **classical/symbolic AI**, specifically *informed* search over a graph, which is typically covered in the same AIML course under "Problem Solving by Search." Including it alongside the ML components lets the report show the *contrast* between learned models and algorithmic planning — and, unlike an uninformed search, A* lets us show the course concept of a **heuristic function** end-to-end, grounded in real GPS data instead of a toy grid.

### 6.1 Representation
The logistics network (road/rail/port/airport checkpoints in `backend/data/graph_topology.json`) is modelled as a `networkx.Graph`, where:
- **Nodes** = checkpoints, each carrying a real GPS `lat`/`lon` (used by the heuristic, §6.3)
- **Edges** = transport legs, weighted simultaneously by `base_time_min` (for a "speed" objective), `cost_inr` (for a "cost" objective), and `risk_weight` (for a "safety" objective)

### 6.2 Algorithm: A* (`f(n) = g(n) + h(n)`)
**InsureRoute's primary route-search algorithm is A\*.** For every node `n` the algorithm considers while expanding the search frontier:

```
f(n) = g(n) + h(n)
```

- **`g(n)`** — the real, accumulated edge weight from the origin to `n`. This is exactly what Dijkstra would accumulate; A* takes no shortcuts here.
- **`h(n)`** — an estimate of the remaining cost from `n` to the destination, used only to decide *which* frontier node to expand next.

The implementation (`find_route_astar()` in `graph_router.py`) calls `networkx.astar_path(G, origin, destination, heuristic=..., weight=...)`, i.e. NetworkX's own A* routine, rather than a hand-rolled priority-queue loop — the priority-queue mechanics are standard and well-tested in NetworkX; the actual intellectual content of this project is the **heuristic design** (§6.3), which is fully custom.

**Why A* over plain Dijkstra:** Dijkstra explores uniformly outward in every direction from the origin because it has no notion of "closer to the goal." A* uses `h(n)` to prioritise nodes that are actually in the geographic direction of the destination. With a *consistent* heuristic it never expands more nodes than Dijkstra (up to tie-breaking), and on large graphs the saving can be substantial; **on this project's 14-node demo graph the measured saving is marginal** (§6.9) — the honest claim is "A* is the correct informed-search framework and never worse", not "A* is dramatically faster here". With an *admissible* heuristic (one that never overestimates the true remaining cost — proved per-objective below), **A\* is still guaranteed to return the optimal path**, exactly like Dijkstra — it is strictly a smarter way to find the same answer, not an approximation.

**When `h(n) = 0`, A\* is mathematically identical to Dijkstra.** The project uses this fact explicitly for the "safety" objective (see below) rather than inventing a heuristic that cannot be proven correct.

### 6.3 The heuristic: GPS-based, and proven admissible per objective
`astar_heuristic()` computes `h(n)` from the great-circle (Haversine) distance between a node's GPS coordinates and the destination's — but **the same distance cannot be reused blindly for every objective**, because "speed," "cost" and "safety" are different units with different admissibility arguments:

| Objective | Edge weight (`g`) | Heuristic `h(n)` | Why it is admissible (never overestimates) |
|---|---|---|---|
| **speed** | `base_time_min` | `haversine_km(n, dest) / MAX_PLAUSIBLE_SPEED_KMH × 60` | Premise (P) below holds with a large margin: the fastest edge in the topology implies 200 km/h (the air leg) versus the 900 km/h (roughly air-freight cruise speed) used in `h`. |
| **cost** | `cost_inr` | `haversine_km(n, dest) × cheapest_₹/km_in_network` | Premise (P) holds with a large margin: `h` uses the network's cheapest rate (sea, ₹8/km, from `multimodal_config.json`) while every edge actually present is charged ≥ ₹18/km (rail 18, road 35, air 220). |
| **safety** | `risk_weight` | `0` | `risk_weight` is **not a spatial quantity** — a short edge can carry more risk than a long one (e.g. a flooded 5 km stretch vs. a clear 50 km highway) — so *no* function of geographic distance can be proven to never overestimate remaining risk. Rather than use a plausible-looking but unprovable heuristic, `h(n) = 0` is used, which makes A* exactly equivalent to Dijkstra for this objective: still correct, just without the geographic speed-up. |

**The exact proof obligation.** Both non-zero heuristics have the form `h(n) = c × haversine_km(n, dest)`. Admissibility *and* consistency follow from the triangle inequality for great-circle distance **provided one per-edge premise holds for the graph data**:

```
(P)   weight(u, v)  ≥  c × haversine_km(u, v)      for every edge (u, v)
```

**(P) is a property of the topology data, not of physics — and it is worth being upfront about this.** The hand-entered `distance_km` in `graph_topology.json` is *smaller* than the true great-circle distance on **6 of the 18 edges** (e.g. `PUNE_DEPOT→CP03` declares 42 km; the coordinates are ~46.7 km apart), so the naive argument "great-circle distance never exceeds the real edge distance" is **not** valid for this dataset. The heuristics are nevertheless safe today because the constants `c` were chosen with deliberate slack (900 km/h vs a 200 km/h fastest edge; ₹8/km vs a ≥₹18/km cheapest edge), so (P) in its *weight* form holds on all 18 edges. This is enforced, not assumed: `HeuristicPremiseTests` fails if (P) is violated on any edge, and separately checks `h(n) ≤` the true Dijkstra remaining cost (admissibility) and `h(x) − h(y) ≤ weight(x, y)` on every edge in both directions (consistency) for **every destination and five cargo/weight graph variants** — 5,576 individual checks in the audit run, 0 violations. If someone later edits the topology so (P) fails, the tests fail rather than A* silently returning a non-optimal route.

> **Explicitly not done:** using raw geographic distance as a heuristic for `cost` or `safety` without this per-objective admissibility argument. The `cost` heuristic required deriving a legitimate lower bound (cheapest ₹/km in the network) rather than just reusing the speed heuristic's constant; the `safety` heuristic could not be made admissible at all given what `risk_weight` currently represents, so it correctly falls back to `h(n)=0` instead of guessing. This distinction — *some* objectives get a real heuristic, one deliberately doesn't — is one of the strongest points to make in a viva, because it shows the heuristic was derived from a correctness argument, not copy-pasted.

### 6.4 Alternative routes: A* is not a k-shortest-path algorithm
`nx.astar_path` — like Dijkstra — returns only **one** optimal path per call. The dashboard needs several ranked alternatives (so the operator can compare options, and so the system has an instant fallback if the top route is later blocked). To keep that behaviour, `find_k_routes_astar()` wraps A* in the same Yen's-algorithm bookkeeping loop that NetworkX's own `shortest_simple_paths` uses internally for Dijkstra — repeatedly re-running the single-path search on a graph view with parts of previously found paths temporarily hidden (`nx.subgraph_view`), and keeping the best of the resulting candidates. **A* is still doing 100% of the actual pathfinding in every one of those calls; the Yen's loop is just deciding which sub-problem to hand it next.** This is stated explicitly here because it is easy to accidentally imply the opposite ("A* finds k shortest paths") — it does not; the k-alternatives mechanism is a separate, well-known algorithm layered on top of it.

### 6.5 Dynamic re-planning (rerouting)
When the ML engine or the live weather feed flags a checkpoint as high-risk, that checkpoint is added to a `blocked_checkpoints` list and the graph is **rebuilt with that node excluded**, then re-searched with A*:

```
Initial route (A* search)
        ↓
New risk/weather information arrives
        ↓
Checkpoint added to blocked_checkpoints
        ↓
Graph rebuilt WITHOUT that node
        ↓
A* search runs again, from scratch, on the new graph
        ↓
New (safer) optimal route returned
```

This is the classic AI notion of **reactive re-planning**: the world model (graph) is updated in response to new sensory evidence (weather/ML risk), and the planner (A*) is re-invoked from scratch rather than trying to patch the old plan. `find_optimal_routes(origin, destination, blocked=[...])` — the same function the Gemini agent's `trigger_reroute` tool calls (§7.2) — is the single entry point for this.

### 6.6 Constraint handling (unchanged, and correctly ordered)
Cargo-specific hard constraints (hazmat chemicals barred from rail, overweight cargo barred from a mode's weight limit, cold-chain cargo barred from unsupported modes) are enforced by `build_constrained_graph()` **before any search runs** — a disallowed edge is simply never added to the graph, rather than being found by A* and discarded afterward. A* therefore only ever sees valid states; constraint satisfaction happens at the state-space construction stage, not the search stage.

### 6.7 Test coverage (empirical verification, not just a correctness argument)
**File:** [`backend/test_graph_router.py`](backend/test_graph_router.py) — 32 tests, standard-library `unittest` (no extra dependency): 31 pass, 1 is skipped when the optional `httpx` package is absent (the HTTP-layer test; the handler-level API tests still run).

Unlike §6.3's admissibility argument (a proof that depends on premise (P)), the test suite checks the premise and the resulting properties **empirically against the real topology**, plus every other claim made above:

| Group | What it checks |
|---|---|
| `HaversineTests` | The distance formula itself: zero self-distance, symmetry, a plausible Pune↔Mumbai bound |
| `HeuristicAdmissibilityTests` | `h(n) ≤ true Dijkstra cost` toward `MUMBAI_DEST` for every reachable node, for both the speed and cost heuristics; confirms the safety heuristic is exactly `0` |
| `HeuristicPremiseTests` | Premise (P) on every edge; admissibility **and consistency** for *every* destination across five cargo/weight variants; safe `h = 0` fallback for unknown nodes |
| `ExpansionComparisonTests` | Total nodes expanded by A* ≤ Dijkstra across all origin/destination pairs (deliberately *not* "strictly fewer" — see §6.9) |
| `EquivalenceWithPreviousAlgorithmTests` | The A*-based alternatives produce the same path set and cost ranking as the previous `nx.shortest_simple_paths` for all three objectives |
| `LivePipelineWiringTests` | Spies on `nx.astar_path` while calling the public `find_optimal_routes()` — proves A* is really executed (initial search, reroute, all three objectives) and that `shortest_simple_paths` is never called |
| `ApiHandlerRegressionTests` | `create_shipment` handler contract, the Gemini `trigger_reroute` tool avoiding a blocked checkpoint, and (if `httpx` is installed) the real HTTP endpoint |
| `AStarBasicCorrectnessTests` | A* returns a specific, hand-traced optimal path (the 90-minute air corridor) and matches `nx.dijkstra_path_length` on cost for all three objectives |
| `ObjectiveTests` | Speed and cost objectives provably select *different* routes (air vs. rail), proving the per-objective weighting is actually wired through |
| `BlockedCheckpointReroutingTests` | Blocking a node removes it from both the direct A* result and the full `find_optimal_routes()` response — the reroute flow in §6.5, exercised end to end |
| `CargoConstraintTests` | Hazmat/overweight edges are absent from the graph *before* search, and never appear in a returned path |
| `AlternativeRoutesTests` | `find_k_routes_astar()` returns multiple, loopless, strictly-ranked paths |
| `RegressionTests` | The public `find_optimal_routes()` response schema, sort order, and empty-result-on-no-path behaviour are unchanged from before the A* migration |

### 6.8 Demonstration logging (for the viva)
`graph_router.py` logs (via Python's standard `logging` module, at `INFO`/`DEBUG` — never sent to the frontend) the origin, destination, priority, which heuristic branch was used, a proxy count of heuristic evaluations (a stand-in for nodes expanded), the final route, and whether the call was a reroute (`blocked` was non-empty). This is deliberately server-side only — normal users only ever see the route JSON — but running the backend with `logging.basicConfig(level=logging.INFO)` makes the A*→reroute sequence in §6.5 directly visible in the console during a live demo.

### 6.9 Measured: nodes expanded, A* vs Dijkstra
Measured on the actual 14-node topology (fmcg, 1 t), counting distinct nodes whose outgoing edges were relaxed, over all 182 reachable origin→destination pairs:

| Objective | Avg. nodes expanded — Dijkstra | Avg. nodes expanded — A* | Pairs where A* expanded fewer / more |
|---|---|---|---|
| speed | 7.00 | 6.88 | 21 / 0 |
| cost | 7.00 | 6.48 | 70 / 0 |
| safety | *(h = 0, identical to Dijkstra by construction)* | — | — |

For the headline request `PUNE_DEPOT → MUMBAI_DEST` the counts are **identical** (speed: 7 vs 7; cost: 9 vs 9). So the honest conclusion is: A* is never worse, saves a little on some pairs, and on this small graph does not visibly change the demo route — its value here is correctness-preserving informed search and a defensible heuristic design; the speed-up would matter on a much larger network (e.g. the all-India 50-hub graph, or real road networks).

---

## 7. Component 4 — Generative & Agentic AI: The Gemini Risk Advisor

**File:** [`backend/core/gemini_agent.py`](backend/core/gemini_agent.py)

### 7.1 Why an LLM is the right tool here (and the others aren't)
Components 1–3 all produce **numbers** (a risk score, an anomaly flag, a path). None of them can, by construction, produce a fluent, context-aware sentence like *"Reroute now — pharmaceutical cargo on ROUTE_B is entering a flood zone; premium will rise 40% if unaddressed."* That is a natural-language synthesis and judgment-call task, which is exactly what a large language model is suited for and a regression/tree model is not. This is a good place in the report to explicitly discuss **why generative AI complements rather than replaces predictive ML** — they solve different sub-problems in the same pipeline.

### 7.2 Architecture: agentic tool-calling loop
This is not a single prompt→response call; it's a **ReAct-style agent loop**:
1. The user's message + shipment context is sent to `gemini-2.5-flash` along with 5 declared tools (`get_disruptions`, `score_route`, `calculate_premium`, `trigger_reroute`, `get_multimodal_options`, `get_weather_forecast`).
2. The model decides, autonomously, whether it needs to *call* one or more of those tools before it can answer (e.g. it can't recommend a premium without calling `calculate_premium` first).
3. Each tool call is executed against the **real backend engines from Components 1 and 3** (not another LLM call) — the ML risk model and the graph router are literally exposed to the LLM as callable functions.
4. Tool outputs are fed back into the same chat session, and the loop repeats (bounded at `max_iterations=5` to guarantee termination) until the model has enough grounded information to produce a final natural-language answer.

This is the key architectural idea to highlight: **the LLM is an orchestrator over deterministic tools, not a source of numeric truth.** The model is explicitly forbidden (by construction, not just by prompt) from inventing a premium or a risk score — it can only report what `calculate_premium`/`score_route` (real code) actually returned. This directly mitigates hallucination risk on the numbers that matter most (money, risk class).

### 7.3 Prompt engineering
The system instruction (`SYSTEM_PROMPT`) encodes explicit **decision rules**, not just a persona:
```
- If any checkpoint risk score > 0.7 → ALWAYS call trigger_reroute proactively
- If route risk > 0.6 AND cargo is pharma/perishable → escalate to all_risk coverage
- If ETA delay > 2 hours → compare multimodal options
- Always call calculate_premium AFTER scoring a route
```
This turns the LLM into something closer to a **rule-guided agent** than a free-form chatbot — a deliberate design choice so its behaviour is predictable and auditable enough for an insurance/logistics context, while still keeping the flexibility of natural language for the parts (explanation, prioritisation) that are hard to hand-code.

### 7.4 Multimodal capability
`analyze_weather_image()` sends a base64 image (e.g. a weather radar screenshot) directly to Gemini 2.5 Flash's vision input alongside the route context, asking it to identify which checkpoints are affected and estimate severity — demonstrating multimodal (vision + text) generative AI, not just text-only chat.

### 7.5 Responsible-AI engineering around the LLM
**File:** [`backend/core/gemini_guard.py`](backend/core/gemini_guard.py)

A thin guard layer wraps every Gemini call with:
- **Prompt-level caching** (5-minute TTL) — avoids re-billing/re-latency for repeated identical questions.
- **Rate limiting** (min 4 seconds between calls, configurable RPM/daily caps) — protects against runaway agentic loops calling the API too fast.
- **Bounded retries with graceful fallback** — if the API is unavailable or the key is unset, the system returns a clearly-labelled fallback message rather than crashing, so the rest of the (deterministic) pipeline keeps working. This is a direct implementation of the architecture's stated resilience principle: *"If Gemini is unavailable, the dashboard still provides data-driven insights"* (see [system_architecture.md](system_architecture.md)).

This is worth a paragraph in any report section on **responsible/production AI deployment** — it shows the LLM is treated as an unreliable, rate-limited, costed external dependency, not a trusted oracle.

---

## 8. Deliberately Non-ML Component: The Actuarial Pricing Engine

**File:** [`backend/core/insurance_engine.py`](backend/core/insurance_engine.py)

The insurance premium is computed by an explicit formula, **not** a learned model:

```
Premium = Cargo Value × Base Rate(mode) × Risk Loading(risk score) × Cargo Multiplier(type) × Coverage Multiplier(plan)
```

This is included in the report as a deliberate **contrast case**: in a domain like insurance pricing, regulators and customers require an explainable, auditable, reproducible calculation — "why was I charged ₹12,400?" must have a formula-based answer, not "the model said so." So even though a regression model *could* be trained to predict premiums end-to-end, the project instead uses ML only where prediction is unavoidable (risk estimation, §4) and keeps the financially/legally sensitive conversion from risk → price as a transparent rule system. This is a real, examinable point about **when to use ML vs. when to use a rule-based/expert system** — a classic AIML syllabus topic in its own right.

---

## 9. Design Choices — Summary Table

| Decision point | Options considered | Chosen | Core reason |
|---|---|---|---|
| Risk estimation task | Classification vs Regression | Regression | Downstream needs a continuous multiplier, not a bucket |
| Risk model | Decision Tree / Random Forest / Linear / Neural Net | Random Forest | Best bias–variance trade-off for ~10 mixed-type features, no scaling needed, interpretable via feature importances |
| Anomaly detection | K-Means / DBSCAN / Isolation Forest / Autoencoder | Isolation Forest | No labels needed, `O(n log n)`, robust without distance metrics |
| Categorical encoding (tree models) | One-Hot vs Label Encoding | Label Encoding | Tree splits are scale/order-invariant; avoids high-dimensional sparse one-hot for low benefit |
| Feature scaling | Needed for Isolation Forest, skipped for Random Forest | Both, situationally | Distance/split-range-sensitive models need it; pure threshold-split trees don't |
| Route planning (single optimal path) | Dijkstra (uninformed) vs A* (informed) | **A\*** | GPS coordinates are available on every node; a heuristic that is admissible/consistent on the actual data (verified by tests, §6.3) lets A* reach the same optimal answer as Dijkstra while never expanding more nodes (the measured saving on the demo graph is marginal, §6.9) |
| Heuristic per objective | One heuristic reused for all objectives vs one derived/justified per objective | Derived per objective (`speed`: time bound, `cost`: rate bound, `safety`: h=0) | Geographic distance is only a valid lower bound for spatial cost units (time, money-per-km); `risk_weight` isn't spatial, so reusing the same heuristic there would be unprovable — h=0 (Dijkstra-equivalent) is used instead |
| Route planning (k alternatives) | `nx.shortest_simple_paths` (Dijkstra-based Yen's) vs Yen's-over-A* | Yen's algorithm with **A\*** as the inner subroutine | Need ranked alternatives for instant fallback, not just one optimum, while still making A* the actual pathfinder — `nx.astar_path` itself is not a k-shortest-path algorithm |
| Constraint handling (hazmat, cold-chain) | Post-search penalty vs pre-search edge removal | Pre-search removal | Keeps search space valid and search logic simple |
| Explanation layer | Hand-coded templates vs LLM | Gemini 2.5 Flash agent | Natural-language synthesis over heterogeneous signals isn't hand-codable at reasonable effort |
| LLM's role | Free-form generation vs grounded tool-calling | Grounded tool-calling agent | Prevents hallucinated numbers on money/risk-critical output |
| Premium calculation | Learned model vs rule-based formula | Rule-based formula | Regulatory/explainability requirement in insurance pricing |
| Training data | Real historical data vs synthetic | Synthetic (domain-distribution-informed) | No public dataset at needed granularity; synthetic generation still encodes real seasonal/diurnal structure |

---

## 10. Limitations (for an honest "Results & Limitations" section)

- **Synthetic ground truth**: both ML models are trained on procedurally generated data whose labels are a known formula (§4.5) or a manually-set contamination prior (§5.3) — reported accuracy will look better than a real deployment would achieve. A future iteration should validate against real, held-out incident logs.
- **No formal train/test split or cross-validation is currently wired into `train_model()`** — worth adding (`train_test_split`, k-fold CV, MAE/RMSE reporting) before citing a numeric accuracy figure in a report; currently only `feature_importances_` is printed.
- **The Isolation Forest module is not currently wired into the live API** (`backend/legacy/`) — it's presented here as an architectural/paradigm artifact, not a production path. State this explicitly if asked "is this running live?"
- **Hyperparameters are fixed/hand-chosen, not tuned** (no grid search / Optuna run) — an easy, well-scoped extension to add for a stronger report ("we additionally ran grid search over `max_depth`, `n_estimators`...").
- **LLM outputs are not evaluated quantitatively** (no held-out prompt/response benchmark) — the report should describe this as qualitative/agentic behaviour validation, not measured accuracy.
- **The A* heuristic's speed constant (900 km/h) is a deliberately conservative, hand-chosen bound**, not derived from a formal analysis of the fleet's actual maximum achievable speed — chosen this way specifically so it stays admissible if the topology changes (§6.3), at some cost to pruning efficiency. A tighter, still-safe bound (e.g. periodically recomputed from the fastest edge actually present) would expand fewer nodes; this is a reasonable "what would you improve" answer in a viva.
- **The A* heuristic's correctness depends on data quality**: the hand-entered `distance_km` values disagree with the GPS coordinates on 6 of 18 edges (§6.3). The heuristics tolerate this only through deliberately loose constants; the fix that would remove the dependency is deriving edge distances from the coordinates (or validating the topology file on load).
- **A* shows almost no speed-up on the 14-node demo graph** (§6.9); do not present it as a performance win — present it as the correct informed-search formulation whose benefit scales with graph size.
- **Two parts of the running system still describe or use Dijkstra, outside the A* pipeline.** (1) The dashboard's "trigger disruption" button is a scripted front-end animation (hard-coded severity `0.89`, text "ISOLATION FOREST ANOMALY DETECTED… DIJKSTRA BYPASS", `InsureRouteDashboard.jsx:632-668`) that makes **no backend routing call** — it only switches to another already-computed route. (2) The legacy endpoints `/route-news` and `/route-intelligence` call `backend/legacy/graph_engine.py::find_route`, which runs `nx.dijkstra_path` on a *separate* 50-hub all-India graph; because the frontend sends core node IDs (`PUNE_DEPOT`, `MUMBAI_DEST`) that do not exist in that graph, it silently substitutes the first two hubs (Madurai→…→Rajahmundry), so those two panels analyse an unrelated path. Neither affects the routes shown on the map, but a viva answer must not claim the whole application uses A*.
- **By contrast with the two ML components above, the A* routing engine's core correctness claim (heuristic admissibility) *is* empirically verified** — `test_graph_router.py` checks it against every reachable node in the real topology, not just argued in comments (§6.7). This asymmetry (tested search algorithm, untested-at-scale ML models) is itself a fair point to raise when discussing the report's rigor.

---

## 11. Future Scope (already reflected in project roadmap docs)

- **Graph Neural Networks** to predict cascading, multi-node failures across the network rather than scoring checkpoints independently — a natural extension once the search-based routing (Component 3) is paired with a learned risk propagation model.
- **Vision-based multimodal intake** — extending §7.4 to ingest live satellite/traffic-camera feeds, not just single uploaded images.
- **Real IoT telemetry** (OBD2 / reefer sensors) replacing the currently simulated disruption feed ([`backend/core/disruption_feed.py`](backend/core/disruption_feed.py)), which would let the ML models in §4–5 finally be validated against ground-truth outcomes instead of synthetic labels.

---

## 12. How to Reproduce

### 12.1 The ML Pipeline

```bash
# 1. Generate the synthetic training dataset (30,000 rows)
cd backend
python scripts/generate_synthetic_data.py

# 2. Train the Random Forest risk model (also runs automatically on first prediction if missing)
python -m core.ml_engine   # or: python -c "from core.ml_engine import train_model; train_model()"

# 3. Inspect feature importances (printed to stdout after training)
```

The trained artifacts (`models/risk_model.pkl`, `models/encoders.pkl`) are what the FastAPI backend loads at inference time via `predict_risk()`.

### 12.2 The A* Routing Tests

```bash
cd backend
python -m unittest test_graph_router -v
```

All tests (§6.7) should pass (the HTTP-layer test is skipped unless `httpx` is installed). To see the demonstration logging (§6.8) during a manual run:

```bash
python -c "
import logging; logging.basicConfig(level=logging.INFO)
from core.graph_router import find_optimal_routes
find_optimal_routes('PUNE_DEPOT', 'MUMBAI_DEST', priority='speed')
find_optimal_routes('PUNE_DEPOT', 'MUMBAI_DEST', priority='speed', blocked=['AP01'])
"
```

The second call blocks the airport hub used by the first call's best route, so the console log shows the reroute sequence (§6.5) end to end: the `REROUTE requested` line, the graph being rebuilt, and a new A* result that avoids `AP01`.

---

## 13. Mapping to Typical AIML Course Outcomes

| Course topic | Demonstrated by |
|---|---|
| Search algorithms (A*, heuristics, admissibility, Dijkstra as a special case) | [`graph_router.py`](backend/core/graph_router.py) — §6 |
| Supervised learning / regression / ensembles | [`ml_engine.py`](backend/core/ml_engine.py) — §4 |
| Unsupervised learning / anomaly detection | [`legacy/model.py`](backend/legacy/model.py) — §5 |
| Feature engineering & preprocessing | [`legacy/preprocessing.py`](backend/legacy/preprocessing.py) — §5.4 |
| Categorical encoding strategies | §4.4 |
| Hyperparameters & bias–variance trade-off | §4.3, §5.3 |
| Synthetic/simulated data generation | [`generate_synthetic_data.py`](backend/scripts/generate_synthetic_data.py) — §4.5 |
| Generative AI / LLMs | [`gemini_agent.py`](backend/core/gemini_agent.py) — §7 |
| Agentic AI / tool use / function calling | §7.2 |
| Prompt engineering | §7.3 |
| Multimodal AI | §7.4 |
| Responsible/production AI (rate limits, fallback, caching) | [`gemini_guard.py`](backend/core/gemini_guard.py) — §7.5 |
| Explainable AI vs black-box models | §8, §9 |
| System/software architecture around AI components | [system_architecture.md](system_architecture.md) — §3 |
| Empirical algorithm verification / testing | [`test_graph_router.py`](backend/test_graph_router.py) — §6.7 |

---

## 14. Tech Stack Reference

| Layer | Technology | Role |
|---|---|---|
| ML | scikit-learn (`RandomForestRegressor`, `IsolationForest`, `LabelEncoder`, `StandardScaler`) | Prediction & anomaly detection |
| Graph/Search | NetworkX (`astar_path`), custom GPS heuristic | A* route planning + Yen's-style ranked alternatives |
| Generative AI | Google Gemini 2.5 Flash (`google-generativeai`) | Natural-language advisory, agentic tool use |
| Backend/serving | FastAPI, Pydantic | API layer, request validation, async orchestration |
| Data | pandas, NumPy | Synthetic data generation, feature engineering |
| Frontend | React 18, Vite, Tailwind, Recharts | Visualisation of AI outputs |
| Deployment | Docker, Google Cloud Run | Serverless hosting for the inference/API layer |

---

## 15. Suggested Slide/Report Structure (if this needs to be a presentation)

1. Problem statement (§2)
2. Architecture overview (§3)
3. **Supervised ML deep-dive** — Random Forest, features, hyperparameters (§4)
4. **Unsupervised ML deep-dive** — Isolation Forest contrast (§5)
5. **Search/planning** — A* re-routing demo (§6)
6. **Agentic GenAI** — live tool-calling trace demo (§7)
7. Why rule-based pricing, not ML (§8) — the "critical thinking" slide
8. Limitations & what you'd do with more time (§10)
9. Live demo (dashboard) + Q&A

---

*This report was compiled directly from the current source in this repository (`backend/core/`, `backend/legacy/`, `backend/scripts/`) — every algorithm, hyperparameter, and formula cited above is copied from the actual implementation, not a marketing description, so it will hold up if an examiner opens the code.*
