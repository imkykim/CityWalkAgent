# CityWalkAgent — Architecture

## Overview

CityWalkAgent models pedestrian experience as a continuous narrative rather than a collection of independent point scores. The design is grounded in Kahneman's dual-process theory: a fast, automatic System 1 evaluates each waypoint as it is encountered; a slower, deliberative System 2 activates at decision-relevant moments to reason about what has been seen and what route to take next. This separation allows the agent to be computationally efficient on unremarkable stretches while investing deeper reasoning at intersections and significant environmental transitions.

---

## System 1: ContinuousAnalyzer

System 1 is responsible for per-waypoint visual scoring. At each waypoint, a Street View image is fetched and passed to a vision-language model (VLM), which evaluates the scene across the active framework's perceptual dimensions (e.g., Safety, Lively, Beautiful, Wealthy in Place Pulse 2.0). The VLM prompt is optionally conditioned on the active persona, reframing evaluation from that persona's perspective.

**3-signal gating.** Not every waypoint triggers a full VLM call. `CognitiveController` applies three signals to decide whether visual analysis is warranted:

1. **Heading delta** — large direction changes (intersections, turns) always trigger analysis.
2. **Cumulative geodesic distance** — analysis is forced after a minimum distance interval regardless of visual change.
3. **Perceptual hash (pHash) distance** — two images are compared; if the scene is visually too similar to the previous analysed frame, the call is skipped.

`CognitiveController` is the single source of truth for pHash computation and the gating decision. Intersections (detected via link count from the Map Tiles API) bypass gating unconditionally — they are always analysed because navigation decisions depend on fresh scores.

Gating is a *sampling strategy*, not a cognitive layer. It controls API cost and latency without affecting the scoring logic.

**Short-term memory (STM).** Every analysed waypoint — its image, scores, reasoning, and metadata — is appended to a rolling STM window. The VLM prompt for each new waypoint includes a condensed summary of recent STM entries, giving System 1 sequential context without the cost of full history.

---

## System 2: Reasoning Pipeline

System 2 activates at two triggers:

- **Intersection** — the navigation graph (Map Tiles API) reports ≥ 3 outgoing links.
- **Significant event** — a score drop beyond a threshold, a dramatic scene change (pHash), or a persona-relevant landmark detected by System 1.

The pipeline runs four stages sequentially. Each stage receives the outputs of all previous stages, so context accumulates:

```
System1Evidence
  → Interpreter   (WHY:  explain score changes, identify barriers/highlights)
  → Decider       (WHAT: branch recommendation, segment avoidance)
  → Planner       (HOW:  optimal continuation, multi-route comparison)
  → Reporter      (OUT:  episode summary → Long-Term Memory)
```

**Interpreter** reads the STM window and produces a structured explanation of why scores changed — distinguishing genuine environmental shifts from noise.

**Decider** uses the Interpreter's output to recommend a direction at the current branch point. At intersections, it fetches candidate Street View images for each outgoing heading and scores them in parallel (Semaphore(48) + retry(2)) before making a soft-priority ranking rather than a hard filter.

**Planner** fetches a walking route from the Google Directions API and computes per-step bearing (`wp_bearing`) toward successive route waypoints. This bearing is fed back into the navigation stack to keep autonomous walks on a coherent path even without explicit turn instructions.

**Reporter** synthesises the episode into a compact narrative unit and writes it to Long-Term Memory (LTM).

---

## Memory

**Short-term memory (STM)** is a fixed-size rolling window (configurable, default ~10 entries) of recent `WaypointAnalysis` objects. It is held in memory only and reset at the start of each route analysis.

**Long-term memory (LTM)** is append-only JSONL storage on disk, partitioned by `agent_id`. It accumulates two kinds of entries:

- **Episodes** — Reporter-generated narrative summaries of route segments, including scores, reasoning, and persona-relevant highlights.
- **Route snapshots** — lightweight records of full route outcomes used for cross-run comparison.

LTM is used by System 2 to detect whether the current environment resembles past routes where specific strategies worked or failed.

---

## Navigation Stack

| API | Role |
|-----|------|
| **Map Tiles API** (Street View session) | Pano graph: `coord → panoId`, link enumeration (headings, intersection detection) |
| **Street View Static API** | Fetches 640×640 images for VLM input and demo display |
| **Maps JavaScript API** | Frontend map rendering and Street View Panorama embedding |
| **Directions API** | Walking route polyline for `wp_bearing` computation in Planner |

The demo frontend fetches the Google Maps JS API key from `/api/maps-key` at load time. VLM credentials are fetched from `/api/config` — no vendor information is hardcoded in the HTML.

---

## Persona Model

Six personas are defined in `src/agent/config/personalities.py`:

| ID | Name |
|----|------|
| `homebuyer` | Prospective Homebuyer |
| `runner` | Urban Runner |
| `parent_with_kids` | Parent with Young Children |
| `photographer` | Street Photographer |
| `elderly` / `elderly_walker` | Elderly Walker |
| `tourist` | Tourist |

Each persona consists of a structured prompt fragment — priorities, sensitivities, and evaluation framing — that is injected into the VLM system prompt before scoring. Persona acts as an *interpretive lens*: the same scene is evaluated through different priorities rather than having scores mechanically post-processed. A deterministic `ScoreTransformer` approach was prototyped and abandoned because it could not capture the qualitative reasoning differences between personas.

Persona also conditions System 2: the Interpreter and Decider are told which persona is active, so route recommendations reflect that persona's preferences.

---

## Concurrency

VLM calls at branch points (Decider) are the primary latency bottleneck. The walk runner issues all candidate-heading VLM calls concurrently under `asyncio.Semaphore(48)` with up to 2 retries per call. Empirical testing (`scripts/test_vlm_concurrency.py`) showed that unlimited `asyncio.gather()` produced ~88% success under load, while the semaphore approach maintained 100% success. Production paths always use the semaphore.

Image downloads (Street View Static API) are parallelised with plain `asyncio.gather()` — no semaphore — since they are not rate-limited in the same way.

---

## Design Principles

- **`CognitiveController` is the single source of truth for pHash.** No other component computes or caches image hashes independently.
- **Intersections are unconditional.** The gating heuristic is bypassed whenever the navigation graph reports a branch point; navigation decisions require fresh scores.
- **Soft priority over hard filtering.** Candidate directions at intersections receive ranked scores rather than binary pass/fail filters, preserving options when the top candidate is only marginally better.
- **Persona is prompt-level, not score-level.** All persona influence flows through the VLM prompt; no arithmetic transformations are applied to raw scores after the fact.
