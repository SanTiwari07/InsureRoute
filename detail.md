# InsureRoute: Comprehensive Technical Deep Dive

**Smart Supply Chain Disruption Detection and Dynamic Insurance Pricing**  
Predictive AI-powered logistics intelligence with real-time weather monitoring, algorithmic rerouting, ML risk scoring, and actuarial-grade dynamic pricing.

---

## 1. Executive Summary

**InsureRoute** is an enterprise-grade predictive supply chain intelligence platform. It fuses machine learning risk scoring, live weather monitoring, Google Gemini AI advisory, multimodal graph routing, and dynamic actuarial pricing into a single unified system.

**The Core Problem:** Global supply chains operate on fragmented, static networks. When disruptions occur, logistics companies face losses not only from delays but also from insurance models that are entirely static — a company pays the same premium whether a shipment crosses a clear highway or a monsoon-flooded zone.

**Key Innovations:**
- **ML-Powered Risk Scoring:** A RandomForest Regressor trained on synthetic transit data predicts per-checkpoint risk scores using weather, time, and road condition inputs.
- **Live Weather Intelligence:** Monitors geographic checkpoints along route corridors via the OpenWeatherMap API.
- **Google Gemini AI Advisory:** Integrates **Gemini 2.5 Flash** (agentic function-calling) for contextual risk assessments, routing decisions, and insurance recommendations.
- **Multimodal Graph Routing:** A NetworkX-based directed graph supporting road, rail, air, and sea transport modes with cargo-aware constraints.
- **Actuarial Dynamic Pricing Engine:** Real-time insurance premium calculation utilizing 5 independent pricing factors.

---

## 2. Actuarial Pricing Engine: Technical Reference

This is the core pricing logic powering real-time dynamic quotations.

### 2.1 Master Formula

```text
Premium (₹) = Cargo Value (₹)
              × Base Rate (by transport mode)
              × Risk Loading Factor (by route risk score & ETA)
              × Cargo Risk Multiplier (by cargo type)
              × Coverage Multiplier (by coverage type)
```
*Minimum premium floor: **₹500***

### 2.2 Factor 1 — Transport Mode Base Rate
Different transport modes possess inherently different base risk profiles.

| Transport Mode | Base Rate | Annual Rate Equivalent |
|---|---|---|
| Rail | **0.15%** of cargo value | Lowest — controlled environment, low theft |
| Air | **0.20%** of cargo value | Low — fast transit, minimal weather exposure |
| Sea | **0.25%** of cargo value | Medium — weather and port risk |
| Road | **0.30%** of cargo value | Highest — traffic, weather, theft exposure |

*For multimodal routes, the base rate is a weighted average.*

### 2.3 Factor 2 — Route Risk Score & Risk Loading
The Route Risk Score is a dynamic `0.0–1.0` index computed in real-time:
```text
Route Risk Score = (ML Score × 0.60) + (Disruption Severity × 0.25) + (Weather Severity × 0.15)
```

This composite score is converted into an exponential **Risk Loading Factor**:

| Risk Score Range | Risk Loading Formula | Risk Class |
|---|---|---|
| 0.0 – 0.29 | `1.0` (no loading) | **LOW** |
| 0.30 – 0.59 | `1.0 + (score − 0.30) × 2.0` | **MEDIUM** (up to 1.6×) |
| 0.60 – 0.79 | `1.6 + (score − 0.60) × 4.0` | **HIGH** (up to 2.4×) |
| 0.80 – 1.0 | `1.6 + (score − 0.60) × 4.0` | **CRITICAL** (up to 3.2×) |

### 2.4 Factor 3 — Cargo Type Multiplier
Different cargo types hold unique risk profiles (fragility, theft attractiveness, temperature sensitivity).

| Cargo Type | Multiplier | Risk Rationale |
|---|---|---|
| Perishables | **1.8×** | Highest — strict temperature control, spoilage risk |
| Pharmaceuticals | **1.6×** | Very high — regulatory requirements, cold chain |
| Chemicals / Industrial| **1.5×** | High — hazmat handling, spillage liability |
| Electronics | **1.4×** | High — extreme theft attractiveness, moisture sensitivity |
| FMCG / Consumer | **1.1×** | Slight surcharge — bulk cargo, moderate theft risk |
| Automotive Parts | **1.0×** | Baseline — standardised, robust cargo |
| Textiles | **0.9×** | Discount — low theft attractiveness, robust |

### 2.5 Factor 4 — Coverage Type Multiplier
| Coverage Type | Multiplier | What's Covered |
|---|---|---|
| Basic | **1.0×** | Named perils only (fire, theft, collision) |
| Comprehensive | **1.6×** | All perils except war and nuclear events |
| All Risk | **2.2×** | True all-risk — maximum protection |

---

## 3. ML Engine: Risk Scoring

The ML risk scoring leverages a highly optimized model to predict delays and disruptions based on environmental and temporal vectors.

### 3.1 Model Architecture
- **Algorithm:** `RandomForestRegressor` (scikit-learn)
- **Hyperparameters:** `n_estimators=100`, `max_depth=12`, `random_state=42`
- **Compute:** Fully parallelized (`n_jobs=-1`)

### 3.2 Input Features
| Feature | Source |
|---|---|
| `hour_of_day` / `day_of_week` | Server Timestamp |
| `weather_code` / `rainfall_mm`| OpenWeatherMap live data |
| `visibility_km` / `wind_speed`| OpenWeatherMap live data |
| `traffic_density` | Time-based heuristics (Rush hour vs Off-peak) |
| `checkpoint_type_enc` | Encoded: road / rail / port / airport |

---

## 4. Routing Engine: NetworkX + A* Search

InsureRoute uses A* search over a NetworkX graph to plot and reroute supply chain topologies instantly when nodes fail.

### 4.1 Graph Topology & Cargo Constraints
- **Topology:** Logistics hubs and transport nodes loaded with GPS coordinates.
- **Constraints:** Heavy logic applied at the edge level, and enforced *before* search runs (the disallowed edge is never added to the graph). For instance, **Chemicals** are blocked from certain modes due to hazmat restrictions, and **Perishables** are restricted on long-duration sea routes.

### 4.2 A* Search: `f(n) = g(n) + h(n)`
InsureRoute's primary route-search algorithm is **A\***, not plain Dijkstra. `g(n)` is the real accumulated cost from the origin; `h(n)` is an estimate of the remaining cost to the destination, derived from the great-circle (Haversine) distance between each node's GPS coordinates:

- **Speed:** minimizes `total_time_min`; `h(n)` = great-circle distance ÷ a conservative maximum plausible speed (900 km/h) — a lower bound on remaining travel time (checked against the topology data by the test suite).
- **Cost:** minimizes `total_cost_inr`; `h(n)` = great-circle distance × the cheapest ₹/km rate in the network — a lower bound on remaining cost (checked against the topology data by the test suite).
- **Safety:** weighs time against the number of transfers and live risk factors; `risk_weight` is not a spatial quantity, so `h(n) = 0` is used instead of an unprovable geographic guess — at `h(n)=0`, A* is mathematically identical to Dijkstra, so correctness is preserved.

Because `nx.astar_path` (like Dijkstra) returns only one optimal path per call, ranked route **alternatives** are produced by re-running A* inside the same Yen's-algorithm bookkeeping loop NetworkX's own `shortest_simple_paths` uses internally — A* remains the algorithm doing the actual pathfinding at every step. See [AIML_PROJECT_REPORT.md](AIML_PROJECT_REPORT.md) §6 for the full admissibility proof and design rationale.

---

## 5. Google Gemini AI Integration

InsureRoute seamlessly implements **Google Gemini 2.5 Flash** via direct REST API and native agentic function calling.

### Agentic Capabilities
The agent acts autonomously by invoking custom internal Python tools:
- `get_disruptions()`
- `score_route()`
- `calculate_premium()`
- `trigger_reroute()`

This allows the AI to formulate highly accurate, context-aware operational advice, turning raw telemetry into natural language dashboards for supply chain operators.

---

## 6. Sustainable Development Goals (SDGs)

| SDG | Contribution |
|---|---|
| **SDG 9 (Industry, Innovation & Infrastructure)** | Builds resilient logistics infrastructure through predictive ML. |
| **SDG 11 (Sustainable Cities & Communities)** | Dynamically reroutes cargo away from hazardous zones, improving community road safety. |
| **SDG 13 (Climate Action)** | Mitigates supply chain disruption via real-time extreme weather integration and minimizes Scope 3 emissions by avoiding congestion. |

---

## 7. Future Roadmap

1. **LLM Multi-Modal Intake:** Using Gemini Vision to analyze satellite imagery and traffic camera feeds in real-time.
2. **Live Edge IoT Telemetry:** Ingesting OBD2 and Thermo King reefer data via Pub/Sub.
3. **Graph Neural Networks (GNNs):** Upgrading the NetworkX topology to predict cascading multi-node failures.
4. **Smart Contracts:** Enabling blockchain-based, automated premium settlements.