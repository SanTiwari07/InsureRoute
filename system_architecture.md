# System Architecture

**Project:** InsureRoute  
**Domain:** Predictive AI, Algorithmic Routing, InsurTech, Google Gemini AI  

---

## 1. High-Level Architecture Overview

InsureRoute is engineered on a strictly decoupled, microservices-inspired architecture. The application separates heavy mathematical processing capabilities (machine learning anomaly detection, graph routing, dynamic pricing, and AI advisory) from the client-side visualisation loop.

This separation ensures:
1. **Computational Stability:** ML matrix operations and Gemini API calls do not block the React render thread.
2. **Scalability:** Each tier can be hosted and scaled independently using serverless technologies like **Google Cloud Run**.
3. **Security:** Proprietary actuarial formulas and API keys remain on the server, exposing only calculated outputs to the browser.
4. **Resilience:** Each subsystem degrades gracefully. If the weather API is offline, ML still works. If Gemini is unavailable, the dashboard still provides data-driven insights.

---

## 2. Infrastructure Tiers

### 2.1. The Client Tier (Frontend)
The user-facing application is built for rendering efficiency and responsive state management.

- **Framework:** React 18 and Vite
- **Design System:** Tailwind CSS with custom design tokens
- **State Management:** React Query for asynchronous data fetching, caching, and synchronization.
- **Visualisation Libraries:** React-Leaflet for dynamic interactive mapping, Recharts, Framer Motion.
- **Role:** Acts as the digital twin. Parses complex JSON from the backend and orchestrates visual state across the Dashboard, KPI trackers, interactive map, insurance panel, weather banner, route timeline, AI advisor panel, news panel, and chronological event log.

### 2.2. The API Gateway Layer (Middleware)
Provides the standardised communication protocol bridging the UI and the computational pipeline.

- **Framework:** FastAPI (Python) deployed on Google Cloud Run.
- **Data Validation:** Pydantic models automatically sanitise and structure incoming payloads.
- **Async Architecture:** Native async/await for non-blocking weather polling and Gemini API calls.
- **Key Endpoints:**
  - `POST /api/v1/shipments/create` — Initiates a new shipment analysis.
  - `GET /api/v1/routes/{id}/options` — Fetches algorithmic route options.
  - `POST /api/v1/insurance/quote` — Generates a dynamic actuarial quote based on live data.
  - `POST /api/v1/gemini/chat` — Direct interface to the Gemini 2.5 Flash agent.
  - `GET /api/v1/weather-status` — Polling endpoint for live OpenWeatherMap status.

### 2.3. The Computational Tier (Backend Engines)
The core mathematical execution environment contains robust, independent modules:

#### A. Predictability Engine (`ml_engine.py`)
- **Algorithm:** `sklearn.ensemble.RandomForestRegressor`
- **Function:** Ingests live telemetry parameters (cargo type, weather variables, geographic timestamps, traffic density). Outputs an anomaly/risk score mapping to current danger levels.

#### B. Weather Intelligence Service (`weather_service.py`)
- **Data Source:** OpenWeatherMap API
- **Monitoring:** Dynamically queries exact GPS coordinates based on the generated A* route path.
- **Classification:** Evaluates real-time rain volume, wind speed, and severe weather codes to trigger automatic dashboard disruptions.

#### C. Algorithmic Routing Engine (`graph_router.py`)
- **Library:** NetworkX
- **Algorithm:** A* search (`f(n) = g(n) + h(n)`), with a GPS-based (Haversine) admissible heuristic for the speed/cost objectives, and `h(n) = 0` (Dijkstra-equivalent) for the safety objective where a geographic heuristic cannot be proven admissible. Ranked alternative routes are produced by re-running A* in a Yen's-algorithm loop.
- **Function:** On anomaly detection or dangerous weather, the affected checkpoint is excluded from the graph and A* is re-run from scratch, returning a safer bypass route.

#### D. Actuarial Pricing Engine (`insurance_engine.py`)
- **Formula:** `Base Rate × Risk Loading × Cargo Multiplier × Coverage Multiplier × Cargo Value`
- **Function:** Converts the live composite risk score (derived from ML, IoT, and weather) into an exponential financial risk loading factor. Calculates dynamic insurance quotes in real-time, providing immediate financial ROI feedback to the user.

#### E. Google Gemini AI Advisor (`gemini_agent.py`)
- **Model:** **Gemini 2.5 Flash**
- **Architecture:** Agentic Function Calling
- **Function:** The agent has access to multiple internal tools (`score_route`, `calculate_premium`, `get_disruptions`). It synthesises anomaly scores, weather data, route topology, and insurance pricing into natural-language risk assessments and automated operational decisions.

---

## 3. Data Flow (Request Lifecycle)

```text
User Action (React Dashboard)
      
      
[Frontend] HTTP POST Payload (Origin, Dest, Cargo Value, Cargo Type)
      
      
[FastAPI on Cloud Run] API Gateway
      
       NetworkX Graph Engine (A* search calculates initial shortest paths)
      
       OpenWeatherMap API (Fetches live coordinate telemetry)
      
       RandomForest ML Model (Scores path risk based on telemetry)
      
       Insurance Engine (Calculates dynamic premium arbitrage)
      
       Gemini 2.5 Flash Agent (Generates contextual natural language insights)
      
      
[Backend] Serializes all outputs into consolidated JSON
      
      
[Frontend] Renders animated routes, populates KPIs, displays PDF quotes
```

---

## 4. Scalability and Cloud Readiness

InsureRoute is designed for serverless cloud execution:

- **Stateless Execution:** The FastAPI backend relies on in-memory computation and external API calls per request, making it perfectly suited for horizontal scaling on **Google Cloud Run**.
- **Edge Delivery:** The compiled Vite/React static assets can be deployed to global CDNs like **Firebase Hosting** for sub-100ms load times worldwide.
- **Asynchronous Design:** Blocking I/O operations (like fetching weather for 10 checkpoints) run in ThreadPoolExecutors, ensuring the event loop remains responsive to UI requests.
