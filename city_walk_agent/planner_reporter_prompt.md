# System 2 Planner & Reporter Refactoring — Claude Code Prompt

## Overview

Refactor the System 2 pipeline to redistribute responsibilities from the bloated `orchestrator.py` into proper S2 modules. Three sequential tasks:

1. **Planner**: Move Directions API navigation logic from orchestrator → Planner (stateful navigator)
2. **Reporter**: Redesign as snapshot generator + episode generator + route-level report generator
3. **Frontend**: Add route report UI panel on walk completion

**Critical principle**: We are NOT enforcing a strict Interpreter → Decider → Planner → Reporter pipeline order. Instead, each module owns a functional domain and can be called at different points during the walk lifecycle.

---

## TASK 1: Planner Refactoring

### What to move

The following functions and inline logic currently live in `src/agent/orchestrator.py` and should move to `src/agent/system2/planner.py`:

**Module-level functions to move (orchestrator.py top-level):**
- `_get_walking_route()` → `Planner.fetch_route()` (async, calls Google Directions API)
- `_get_wp_bearing()` → `Planner.get_wp_bearing()` (static/internal)
- `_get_urgency_tier()` → `Planner.get_urgency_tier()` (static)

**Keep in orchestrator as generic geometry utils** (they're used broadly):
- `_calc_bearing()`, `_bearing_to_cardinal()`, `_closest_link()`, `_angle_diff()` — these stay in orchestrator.py as module-level helpers

**Inline logic in `autonomous_walk()` to extract:**
The Directions API state management block near the top of `autonomous_walk()`:
```python
route_waypoints: List[Tuple[float, float]] = []
route_wp_index: int = 0
REROUTE_THRESHOLD_M = 80
# ... _get_walking_route call ...
```
And the per-step navigation context computation inside the loop:
```python
if route_waypoints:
    wp_bearing, route_wp_index, nearest_wp_dist = _get_wp_bearing(...)
    if nearest_wp_dist >= REROUTE_THRESHOLD_M:
        # re-route logic...
```

### New Planner design

```python
# src/agent/system2/planner.py
class Planner:
    """Stateful navigation manager — owns Directions API route and bearing computation."""

    def __init__(self, framework_id: str):
        self.framework_id = framework_id
        self.route_waypoints: List[Tuple[float, float]] = []
        self.wp_index: int = 0
        self.reroute_threshold_m: float = 80.0
        self.reroute_count: int = 0
        self.total_route_distance_m: float = 0.0
        self.logger = get_logger(self.__class__.__name__)

    async def init_route(self, start: Tuple[float, float], dest: Tuple[float, float], api_key: str):
        """Call Directions API at walk start. Store decoded polyline waypoints.
        On failure, log warning and leave route_waypoints empty (fallback to dest_bearing)."""

    def get_navigation_context(self, current_lat, current_lng, dest_lat, dest_lng) -> Dict[str, Any]:
        """Per-step navigation info. Returns:
        {
            "wp_bearing": float | None,       # bearing to next route waypoint
            "dest_bearing": float,            # direct bearing to destination
            "dest_cardinal": str,             # "NE", "S", etc.
            "dist_to_dest_m": float,
            "urgency_tier": str,              # "explore"|"navigate"|"converge"
            "nearest_wp_dist_m": float|None,  # distance to nearest route waypoint
            "dest_context": str,              # formatted string for branch_decision prompt
            "has_route": bool,                # whether Directions API route is available
        }
        Also internally advances wp_index as agent progresses."""

    async def check_and_reroute(self, current_lat, current_lng, dest_lat, dest_lng, api_key: str) -> bool:
        """Check if nearest_wp_dist exceeds threshold; if so, re-fetch route.
        Returns True if re-routed. Increments self.reroute_count."""

    def get_summary(self) -> Dict[str, Any]:
        """Navigation summary for route-level Reporter. Returns:
        {
            "reroute_count": int,
            "total_waypoints_in_route": int,
            "wp_index_reached": int,
            "route_coverage_pct": float,   # wp_index / total_waypoints
            "has_route": bool,
        }"""

    # Keep the existing plan() and route-level stubs — they raise NotImplementedError
    # and are only used by synthesize_route() which is not called yet.
    # Don't delete them, just leave them as-is.
```

### Orchestrator changes

In `autonomous_walk()`:
1. Replace the Directions API init block with:
   ```python
   await self.planner.init_route((start_lat, start_lng), (dest_lat, dest_lng), settings.google_maps_api_key)
   ```
2. Replace per-step nav computation with:
   ```python
   nav = self.planner.get_navigation_context(current_lat, current_lng, dest_lat, dest_lng)
   # Use nav["wp_bearing"], nav["dest_context"], nav["urgency_tier"], etc.
   ```
3. Replace re-route block with:
   ```python
   if nav["nearest_wp_dist_m"] and nav["nearest_wp_dist_m"] >= self.planner.reroute_threshold_m:
       await self.planner.check_and_reroute(current_lat, current_lng, dest_lat, dest_lng, settings.google_maps_api_key)
       nav = self.planner.get_navigation_context(current_lat, current_lng, dest_lat, dest_lng)
   ```
4. In the final result dict, add:
   ```python
   "planner_summary": self.planner.get_summary(),
   ```
5. Delete `_get_walking_route`, `_get_wp_bearing`, `_get_urgency_tier` from orchestrator.py module level.
6. The `Planner` property on `CityWalkAgent` already exists (lazy-loaded). Make sure `init_route` is called **once** at the start of `autonomous_walk()`, and the planner instance is reused throughout the walk.

### Important: Planner is lazy-loaded

The existing `CityWalkAgent.planner` property already lazy-loads a `Planner` instance. Don't create a new one — use `self.planner`. But you need to add the state-reset logic so that each `autonomous_walk()` call starts fresh:

```python
# At start of autonomous_walk:
self.planner.reset()  # Clear state from any previous walk
await self.planner.init_route(...)
```

Add a `reset()` method to Planner that clears route_waypoints, wp_index, reroute_count.

### branch_decision() updates

In `branch_decision()`, the `wp_bearing` parameter is already passed from orchestrator. Now it comes from `nav["wp_bearing"]` instead of inline computation. No changes needed inside `branch_decision()` itself.

---

## TASK 2: Reporter Refactoring

### Current state to change

1. **`MemoryManager._generate_snapshot()`** → Move to `Reporter.generate_snapshot()`
2. **`MemoryManager.update_with_system2_result()` dict construction** → Move to `Reporter.generate_episode()`
3. **Current `Reporter.report_waypoint()`** → Replace with `generate_episode()` (the old waypoint recommendation has no practical use since branch decisions happen before S2)
4. **Current `Reporter.report()`** → Replace with `report_route()`
5. **`Reporter._report_fallback()`** → Update for episode fallback

### New Reporter design

```python
# src/agent/system2/reporter.py
import statistics
from typing import Any, Dict, List, Optional, Tuple
from src.utils.logging import get_logger
from .llm_client import call_llm

class Reporter:
    """Record & Report — generates snapshots, episodes, and route reports.
    
    Three phases, called at different points in the walk lifecycle:
    - generate_snapshot(): Before S2 pipeline (per trigger) — no LLM
    - generate_episode(): After S2 pipeline (per trigger) — LLM 1 call
    - report_route(): After walk completes (once) — LLM 1 call
    """

    def __init__(self, framework_id: str):
        self.framework_id = framework_id
        self.logger = get_logger(self.__class__.__name__)

    # ================================================================
    # Phase 1: Snapshot generation (before Interpreter)
    # ================================================================

    def generate_snapshot(
        self,
        stm_context: Dict[str, Any],
        last_snapshot_waypoint: int,
        current_waypoint_id: int,
    ) -> Dict[str, Any]:
        """Generate route snapshot from STM data since last snapshot.
        
        Pure statistics — no LLM call. Moved from MemoryManager._generate_snapshot().
        
        Returns:
            {
                "span_start": int,
                "span_end": int,
                "trend": "stable"|"improving"|"declining"|"volatile",
                "avg": {dim: float},
                "trajectory": float,       # score delta from span start to end
                "worst_dimension": str|None,
                "barrier_segments": [(start_wp, end_wp), ...],
            }
        """
        # Move the EXACT logic from MemoryManager._generate_snapshot() here.
        # The implementation is a pure function of stm_context + last_snapshot_waypoint + current_waypoint_id.
        # Copy it verbatim, then delete _generate_snapshot from MemoryManager.

    # ================================================================
    # Phase 2: Episode generation (after Interpreter + Decider)
    # ================================================================

    def generate_episode(
        self,
        waypoint_id: int,
        interpretation: Dict[str, Any],
        decision: Dict[str, Any],
        system1_scores: Dict[str, float],
        personality: Any,
        trigger_reason: Any,
        snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate structured episode record from S2 pipeline results.
        
        Single LLM call to produce narrative + significance_tag.
        
        Args:
            waypoint_id: Current waypoint
            interpretation: From Interpreter.interpret_waypoint()
            decision: From Decider.decide_waypoint()
            system1_scores: Raw S1 scores
            personality: Agent personality
            trigger_reason: Why S2 was triggered
            snapshot: The snapshot generated in Phase 1 for this trigger
            
        Returns:
            {
                "waypoint_id": int,
                "significance": str,            # from Decider
                "avoid": bool,                  # from Decider
                "interpretation": str,          # from Interpreter
                "key_concern": str|None,        # from Interpreter
                "score_change_reason": str|None,# from Interpreter
                "scores": {dim: float},
                "trigger_reason": str,
                "narrative": str,               # NEW: LLM-generated 1-2 sentence narrative
                "significance_tag": str,        # NEW: "transition"|"barrier"|"highlight"|"routine"
                "persona_reaction": str,        # NEW: persona perspective one-liner for UI
                "snapshot_trend": str,          # trend from the associated snapshot
            }
        """
        # 1. Build the structured fields from inputs (no LLM needed for these):
        episode = {
            "waypoint_id": waypoint_id,
            "significance": decision.get("significance", "medium"),
            "avoid": decision.get("avoid", False),
            "interpretation": interpretation.get("text", ""),
            "key_concern": interpretation.get("key_concern"),
            "score_change_reason": interpretation.get("score_change_reason"),
            "scores": system1_scores,
            "trigger_reason": getattr(trigger_reason, "value", str(trigger_reason)) if trigger_reason else "unknown",
            "snapshot_trend": snapshot.get("trend", "stable"),
        }

        # 2. LLM call for narrative + significance_tag + persona_reaction
        persona_name = getattr(personality, "name", "Unknown")
        avg_score = sum(system1_scores.values()) / len(system1_scores) if system1_scores else 5.0
        
        prompt = f"""You are recording a walking experience episode for a specific persona.

Persona: {persona_name}
Waypoint: {waypoint_id}
Average score: {avg_score:.1f}/10
Trend in this segment: {snapshot.get("trend", "stable")}
Interpretation: {interpretation.get("text", "")}
Key concern: {interpretation.get("key_concern", "none")}
Decision: {"AVOID" if decision.get("avoid") else "ACCEPTABLE"} (significance: {decision.get("significance", "medium")})

Generate a concise episode record. Respond ONLY with valid JSON:
{{
  "narrative": "1-2 sentences describing this moment in the walk, written as a walking journal entry",
  "significance_tag": "one of: transition, barrier, highlight, routine",
  "persona_reaction": "max 15 chars, first-person casual reaction from the persona, e.g. 'Love this path!' or '여기 위험해...'"
}}"""

        result = call_llm(prompt, max_tokens=200)
        if result:
            episode["narrative"] = result.get("narrative", "")
            episode["significance_tag"] = result.get("significance_tag", "routine")
            episode["persona_reaction"] = result.get("persona_reaction", "")
        else:
            # Fallback
            episode["narrative"] = interpretation.get("text", "")[:100]
            episode["significance_tag"] = "barrier" if decision.get("avoid") else "routine"
            episode["persona_reaction"] = ""

        return episode

    # ================================================================
    # Phase 3: Route-level report (after walk completes)
    # ================================================================

    def report_route(
        self,
        snapshots: List[Dict[str, Any]],
        episodes: List[Dict[str, Any]],
        route_stats: Dict[str, Any],
        planner_summary: Dict[str, Any],
        personality: Any,
    ) -> Dict[str, Any]:
        """Generate comprehensive route report from accumulated data.
        
        Single LLM call. Called once at end of autonomous_walk().
        
        Args:
            snapshots: All snapshots generated during walk (Reporter.generate_snapshot outputs)
            episodes: All episodes generated during walk (Reporter.generate_episode outputs)
            route_stats: Basic statistics {steps, analyzed_steps, skip_rate, arrived, 
                         final_distance_m, duration_seconds, dimension_avgs}
            planner_summary: From Planner.get_summary() {reroute_count, route_coverage_pct, ...}
            personality: Agent personality
            
        Returns:
            {
                "executive_summary": str,       # 3-4 sentences, persona perspective
                "score_narrative": str,         # Score trend story
                "segment_highlights": [         # Top 2-3 notable moments from episodes
                    {"waypoint_id": int, "type": str, "description": str}
                ],
                "dimension_verdicts": {         # Per-dimension one-line verdict
                    "safety": str, "lively": str, "beautiful": str, "wealthy": str
                },
                "recommendation": str,          # "recommended"|"acceptable"|"avoid"
                "recommendation_reason": str,
                "navigation_note": str,         # Planner summary sentence
                "persona_suitability": str,     # Why this route fits/doesn't fit this persona
            }
        """
        persona_name = getattr(personality, "name", "Unknown")
        persona_desc = getattr(personality, "description", "")

        # Compress snapshots into a concise table
        snapshot_lines = []
        for s in snapshots:
            traj = s.get("trajectory", 0.0)
            traj_str = f"+{traj:.1f}" if traj >= 0 else f"{traj:.1f}"
            avg = s.get("avg", {})
            dim_str = " ".join(f"{k}={v:.1f}" for k, v in avg.items())
            barrier = " ⚠barrier" if s.get("barrier_segments") else ""
            snapshot_lines.append(
                f"WP{s['span_start']}-{s['span_end']}: {s['trend'].upper()} {traj_str} [{dim_str}]{barrier}"
            )
        snapshots_text = "\n".join(snapshot_lines) if snapshot_lines else "No snapshots recorded."

        # Compress episodes — prioritize high significance, max 5
        sorted_eps = sorted(episodes, key=lambda e: {"high": 0, "medium": 1, "low": 2}.get(e.get("significance", "low"), 2))
        top_episodes = sorted_eps[:5]
        episode_lines = []
        for ep in top_episodes:
            avoid_str = " ⚠AVOID" if ep.get("avoid") else ""
            episode_lines.append(
                f"WP{ep['waypoint_id']} [{ep.get('significance_tag','routine').upper()}{avoid_str}]: "
                f"{ep.get('narrative', ep.get('interpretation', ''))[:80]}"
            )
        episodes_text = "\n".join(episode_lines) if episode_lines else "No episodes recorded."

        # Route statistics
        stats_text = (
            f"Steps: {route_stats.get('steps', '?')} (analyzed: {route_stats.get('analyzed_steps', '?')}, "
            f"skip rate: {route_stats.get('skip_rate', 0):.0%})\n"
            f"Arrived: {'Yes' if route_stats.get('arrived') else 'No'} "
            f"(final distance: {route_stats.get('final_distance_m', '?')}m)\n"
            f"Duration: {route_stats.get('duration_seconds', 0):.0f}s"
        )

        # Dimension averages
        dim_avgs = route_stats.get("dimension_avgs", {})
        dim_avg_text = " ".join(f"{k}={v:.1f}" for k, v in dim_avgs.items()) if dim_avgs else "N/A"

        # Planner summary
        nav_text = (
            f"Re-routes: {planner_summary.get('reroute_count', 0)}, "
            f"Route coverage: {planner_summary.get('route_coverage_pct', 0):.0%}, "
            f"Has Directions route: {'Yes' if planner_summary.get('has_route') else 'No'}"
        )

        prompt = f"""You are writing a comprehensive walking route evaluation report for a specific persona.

Persona: {persona_name}
Description: {persona_desc}

== Route Statistics ==
{stats_text}
Overall dimension averages: {dim_avg_text}

== Navigation ==
{nav_text}

== Segment Snapshots ==
{snapshots_text}

== Key Episodes ==
{episodes_text}

Write a comprehensive evaluation. Respond ONLY with valid JSON:
{{
  "executive_summary": "3-4 sentences: overall impression of this walk from the persona's perspective",
  "score_narrative": "2-3 sentences: how the walking experience quality changed throughout the route",
  "segment_highlights": [
    {{"waypoint_id": <int>, "type": "highlight|barrier|transition", "description": "one sentence"}}
  ],
  "dimension_verdicts": {{
    "safety": "one sentence verdict",
    "lively": "one sentence verdict",
    "beautiful": "one sentence verdict",
    "wealthy": "one sentence verdict"
  }},
  "recommendation": "recommended|acceptable|avoid",
  "recommendation_reason": "one sentence",
  "navigation_note": "one sentence about the navigation experience",
  "persona_suitability": "one sentence: why this route is or isn't suitable for this persona"
}}"""

        self.logger.info("Generating route-level report via LLM")
        result = call_llm(prompt, max_tokens=800)

        if result and "executive_summary" in result:
            self.logger.info(f"Route report generated: recommendation={result.get('recommendation')}")
            return result

        # Fallback: heuristic report
        self.logger.warning("Route report LLM failed — using fallback")
        return self._report_route_fallback(route_stats, dim_avgs, snapshots, episodes)

    def _report_route_fallback(
        self,
        route_stats: Dict,
        dim_avgs: Dict,
        snapshots: List[Dict],
        episodes: List[Dict],
    ) -> Dict[str, Any]:
        """Heuristic fallback when LLM fails for route report."""
        overall_avg = sum(dim_avgs.values()) / len(dim_avgs) if dim_avgs else 5.0
        rec = "recommended" if overall_avg >= 7 else "acceptable" if overall_avg >= 5 else "avoid"
        barrier_count = sum(1 for e in episodes if e.get("avoid"))
        
        return {
            "executive_summary": f"Walk completed in {route_stats.get('steps', '?')} steps. Average score: {overall_avg:.1f}/10.",
            "score_narrative": f"The route showed {'consistent' if not snapshots else snapshots[-1].get('trend', 'stable')} quality.",
            "segment_highlights": [
                {"waypoint_id": e["waypoint_id"], "type": e.get("significance_tag", "routine"),
                 "description": e.get("narrative", "")[:60]}
                for e in episodes[:3]
            ],
            "dimension_verdicts": {dim: f"Average: {score:.1f}/10" for dim, score in dim_avgs.items()},
            "recommendation": rec,
            "recommendation_reason": f"Overall average {overall_avg:.1f}/10 with {barrier_count} barrier segments.",
            "navigation_note": f"Route completed with {route_stats.get('steps', '?')} steps.",
            "persona_suitability": "See dimension verdicts for details.",
        }
```

### MemoryManager changes

1. **Delete `_generate_snapshot()` method** from `MemoryManager`.
2. **Simplify `prepare_context_for_reasoning()`**: Remove the snapshot generation block. It should only prepare stm_context and ltm_patterns (existing snapshots + episodes). The snapshot for the current trigger is now generated by Reporter in PersonaReasoner.reason().
3. **Simplify `update_with_system2_result()`**: Instead of building the episode dict manually from ReasoningResult attributes, accept the pre-built episode dict from Reporter:

```python
def update_with_system2_result(
    self, waypoint_id: int, reasoning_result: Any, episode: Dict[str, Any]
) -> None:
    """Update memory with S2 result. Episode is pre-built by Reporter."""
    self._system2_results[waypoint_id] = reasoning_result
    self._route_reasoning_log.append(episode)
    # ... logging ...
```

### PersonaReasoner changes

Update `PersonaReasoner.reason()` to use the new Reporter flow:

```python
def reason(self, waypoint_id, trigger_reason, ..., stm_context, ltm_patterns, ...):
    start = time.time()
    
    # Phase 1: Reporter generates snapshot FIRST
    snapshot = self.reporter.generate_snapshot(
        stm_context=stm_context,
        last_snapshot_waypoint=self._last_snapshot_waypoint,
        current_waypoint_id=waypoint_id,
    )
    self._last_snapshot_waypoint = waypoint_id
    
    # Add current snapshot to ltm_patterns for Interpreter/Decider
    if ltm_patterns is None:
        ltm_patterns = {"snapshots": [], "reasoning_episodes": []}
    ltm_patterns = {**ltm_patterns}  # shallow copy
    ltm_patterns["snapshots"] = list(ltm_patterns.get("snapshots", [])) + [snapshot]
    
    # S2 Pipeline
    interpretation = self.interpreter.interpret_waypoint(
        ..., ltm_patterns=ltm_patterns, ...
    )
    decision = self.decider.decide_waypoint(
        ..., ltm_patterns=ltm_patterns, ...
    )
    
    # Phase 2: Reporter generates episode
    episode = self.reporter.generate_episode(
        waypoint_id=waypoint_id,
        interpretation=interpretation,
        decision=decision,
        system1_scores=system1_scores,
        personality=personality,
        trigger_reason=trigger_reason,
        snapshot=snapshot,
    )
    
    # Build ReasoningResult (keep existing structure for compatibility)
    result = ReasoningResult(
        waypoint_id=waypoint_id,
        trigger_reason=trigger_reason,
        interpretation=interpretation.get("text", ""),
        score_change_reason=interpretation.get("score_change_reason"),
        persona_divergence=interpretation.get("persona_divergence"),
        key_concern=interpretation.get("key_concern"),
        significance=decision.get("significance", "medium"),
        avoid_recommendation=decision.get("avoid", False),
        decision_reason=decision.get("reason"),
        prediction=None,
        alternative_suggestion=None,
        recommendation=episode.get("narrative"),  # was report["message"]
        confidence=float(decision.get("confidence", 0.5)),
        # ... rest same as current ...
    )
    
    self.reasoning_history.append(result)
    
    # Return both result and the new artifacts
    # Store snapshot and episode on the result for the caller to pass to MemoryManager
    result._snapshot = snapshot
    result._episode = episode
    
    return result
```

**Important**: The PersonaReasoner currently tracks `_last_snapshot_waypoint` — this needs to be moved to PersonaReasoner from MemoryManager since Reporter now generates snapshots through PersonaReasoner. Actually, cleaner: have PersonaReasoner hold this state since it already tracks reasoning_history.

Add to PersonaReasoner `__init__`:
```python
self._last_snapshot_waypoint: int = 0
```

### Orchestrator S2 trigger block update

In `autonomous_walk()`, where S2 reasoning is triggered (inside the `if should_reason and context:` block), update to:

```python
if should_reason and context:
    try:
        reasoning_result = self.persona_reasoner.reason(
            waypoint_id=...,
            trigger_reason=context["trigger_reason"],
            current_image_path=context["image_path"],
            system1_scores=analysis.persona_scores,
            system1_reasoning=analysis.persona_reasoning,
            stm_context=context["stm_context"],
            ltm_patterns=context.get("ltm_patterns"),
            personality=self.personality,
            route_metadata=route_meta,
            waypoints_since_trigger=context.get("waypoints_since_trigger", 0),
        )
        
        # Extract Reporter artifacts from result
        snapshot = getattr(reasoning_result, "_snapshot", None)
        episode = getattr(reasoning_result, "_episode", None)
        
        # Update memory with pre-built episode
        if snapshot:
            memory_manager._route_snapshots.append(snapshot)
        memory_manager.update_with_system2_result(
            waypoint_id=analysis.waypoint_id,
            reasoning_result=reasoning_result,
            episode=episode or {},  # fallback to empty dict
        )
    except Exception as e:
        self.logger.warning(f"S2 reasoning failed: {e}")
```

### Also update: MemoryManager.prepare_context_for_reasoning()

Remove the snapshot generation block. Before (current code):
```python
snapshot = self._generate_snapshot(waypoint_analysis.waypoint_id)
self._route_snapshots.append(snapshot)
self._last_snapshot_waypoint = waypoint_analysis.waypoint_id
```

After: Just delete those 3 lines and the associated logging. The ltm_patterns in the context should still include existing `_route_snapshots` — just not generate a new one.

---

## TASK 3: Route-level Report + Frontend UI

### Backend: autonomous_walk() completion

After the walk loop ends in `autonomous_walk()`, before building the final result dict, add:

```python
# Generate route-level report
route_report = None
try:
    # Compute dimension averages from route_taken
    all_scores = [step["scores"] for step in route_taken if step.get("scores")]
    dim_avgs = {}
    if all_scores:
        dims = list(all_scores[0].keys())
        for dim in dims:
            vals = [s[dim] for s in all_scores if dim in s]
            dim_avgs[dim] = round(sum(vals) / len(vals), 2) if vals else 0.0

    route_report = self.reporter.report_route(
        snapshots=memory_manager._route_snapshots,
        episodes=memory_manager._route_reasoning_log,
        route_stats={
            "steps": len(route_taken),
            "analyzed_steps": analyzed_step_count,
            "skip_rate": round(1 - analyzed_step_count / max(step + 1, 1), 2),
            "arrived": arrived,
            "final_distance_m": round(final_dist, 1),
            "duration_seconds": round(time.time() - walk_start_ts, 1),
            "dimension_avgs": dim_avgs,
        },
        planner_summary=self.planner.get_summary(),
        personality=self.personality,
    )
    self.logger.info(f"Route report generated: {route_report.get('recommendation', '?')}")
except Exception as e:
    self.logger.warning(f"Route report generation failed: {e}")

# Send route_report via SSE before "complete"
if step_callback and route_report:
    await step_callback({"__event__": "route_report", **route_report})
```

Add `"route_report": route_report` to the final result dict as well.

### Server: SSE event type

In `demo/server.py`, the `step_callback` already handles `__event__` extraction:
```python
async def step_callback(step_result: dict):
    event = step_result.pop("__event__", None)
    if event:
        await q.put({"type": event, "data": step_result})
```
So `route_report` event type is automatically handled. No server changes needed.

Also update the `complete` event data to include the report:
```python
await q.put({"type": "complete", "data": {
    "arrived": result.get("arrived"),
    "steps": result.get("steps"),
    "final_distance_m": result.get("final_distance_m"),
    "persona": result.get("persona"),
    "route_report": result.get("route_report"),  # ADD THIS
    "memory_debug": { ... },
}})
```

### Frontend: Route Report Panel UI

In `demo/index.html`, add handling for the `route_report` SSE event and a summary panel.

**1. Add CSS for the report panel** (add near the existing `.speech-bubble` styles):

```css
/* ── Route Report Panel ── */
#route-report-overlay {
  display: none;
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(0,0,0,0.5);
  z-index: 9999;
  justify-content: center;
  align-items: center;
}
#route-report-overlay.active { display: flex; }

#route-report-panel {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 24px;
  max-width: 520px;
  width: 90vw;
  max-height: 80vh;
  overflow-y: auto;
  box-shadow: 0 8px 32px rgba(0,0,0,0.2);
  position: relative;
}

#route-report-panel .rr-close {
  position: absolute; top: 12px; right: 16px;
  background: none; border: none; font-size: 18px;
  color: var(--text-3); cursor: pointer;
}
#route-report-panel .rr-close:hover { color: var(--text-1); }

#route-report-panel .rr-title {
  font-size: 16px; font-weight: 700; color: var(--text-1);
  margin-bottom: 4px;
}
#route-report-panel .rr-persona {
  font-size: 11px; color: var(--text-3); margin-bottom: 16px;
}

#route-report-panel .rr-badge {
  display: inline-block; padding: 3px 10px; border-radius: 12px;
  font-size: 11px; font-weight: 600; margin-bottom: 12px;
}
.rr-badge.recommended { background: #d1fae5; color: #065f46; }
.rr-badge.acceptable  { background: #fef3c7; color: #92400e; }
.rr-badge.avoid        { background: #fee2e2; color: #991b1b; }

#route-report-panel .rr-section {
  margin-bottom: 14px;
}
#route-report-panel .rr-section-title {
  font-size: 11px; font-weight: 600; color: var(--text-3);
  text-transform: uppercase; letter-spacing: 0.5px;
  margin-bottom: 4px;
}
#route-report-panel .rr-text {
  font-size: 13px; color: var(--text-1); line-height: 1.5;
}

#route-report-panel .rr-dim-grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 6px;
}
#route-report-panel .rr-dim-item {
  background: var(--surface); border-radius: 8px; padding: 8px 10px;
}
#route-report-panel .rr-dim-label {
  font-size: 10px; font-weight: 600; color: var(--text-3);
  text-transform: uppercase; margin-bottom: 2px;
}
#route-report-panel .rr-dim-verdict {
  font-size: 12px; color: var(--text-1); line-height: 1.4;
}

#route-report-panel .rr-highlights {
  list-style: none; padding: 0; margin: 0;
}
#route-report-panel .rr-highlights li {
  font-size: 12px; color: var(--text-1); padding: 4px 0;
  border-bottom: 1px solid var(--border);
}
#route-report-panel .rr-highlights li:last-child { border-bottom: none; }
.rr-hl-tag {
  display: inline-block; font-size: 9px; font-weight: 600;
  padding: 1px 5px; border-radius: 4px; margin-right: 4px;
}
.rr-hl-tag.highlight  { background: #d1fae5; color: #065f46; }
.rr-hl-tag.barrier    { background: #fee2e2; color: #991b1b; }
.rr-hl-tag.transition { background: #dbeafe; color: #1e40af; }
.rr-hl-tag.routine    { background: #f3f4f6; color: #6b7280; }
```

**2. Add HTML** (add right before `</body>`):

```html
<div id="route-report-overlay">
  <div id="route-report-panel">
    <button class="rr-close" onclick="closeRouteReport()">&times;</button>
    <div class="rr-title">Route Report</div>
    <div class="rr-persona" id="rr-persona"></div>
    <div id="rr-badge-area"></div>
    <div id="rr-body"></div>
  </div>
</div>
```

**3. Add JS handler** (add near the SSE handler section):

```javascript
// ═══════════════════════════════════════════════════════
// ROUTE REPORT
// ═══════════════════════════════════════════════════════

function handleRouteReport(data) {
  log('Route report received: ' + (data.recommendation || '?'), 'ok');
  renderRouteReport(data);
}

function renderRouteReport(r) {
  const overlay = document.getElementById('route-report-overlay');
  const persona = document.getElementById('rr-persona');
  const badgeArea = document.getElementById('rr-badge-area');
  const body = document.getElementById('rr-body');

  persona.textContent = currentPersona;

  // Recommendation badge
  const rec = r.recommendation || 'acceptable';
  const recLabel = rec === 'recommended' ? '✓ Recommended' : rec === 'avoid' ? '✗ Avoid' : '— Acceptable';
  badgeArea.innerHTML = `<span class="rr-badge ${rec}">${recLabel}</span>`;

  let html = '';

  // Executive summary
  if (r.executive_summary) {
    html += `<div class="rr-section">
      <div class="rr-section-title">Summary</div>
      <div class="rr-text">${escHtml(r.executive_summary)}</div>
    </div>`;
  }

  // Score narrative
  if (r.score_narrative) {
    html += `<div class="rr-section">
      <div class="rr-section-title">Experience</div>
      <div class="rr-text">${escHtml(r.score_narrative)}</div>
    </div>`;
  }

  // Dimension verdicts
  if (r.dimension_verdicts && Object.keys(r.dimension_verdicts).length) {
    let dimHtml = '<div class="rr-dim-grid">';
    for (const [dim, verdict] of Object.entries(r.dimension_verdicts)) {
      dimHtml += `<div class="rr-dim-item">
        <div class="rr-dim-label">${escHtml(dim)}</div>
        <div class="rr-dim-verdict">${escHtml(verdict)}</div>
      </div>`;
    }
    dimHtml += '</div>';
    html += `<div class="rr-section">
      <div class="rr-section-title">Dimensions</div>
      ${dimHtml}
    </div>`;
  }

  // Segment highlights
  if (r.segment_highlights && r.segment_highlights.length) {
    let hlHtml = '<ul class="rr-highlights">';
    for (const hl of r.segment_highlights) {
      const tag = hl.type || 'routine';
      hlHtml += `<li>
        <span class="rr-hl-tag ${tag}">${tag}</span>
        WP${hl.waypoint_id}: ${escHtml(hl.description || '')}
      </li>`;
    }
    hlHtml += '</ul>';
    html += `<div class="rr-section">
      <div class="rr-section-title">Key Moments</div>
      ${hlHtml}
    </div>`;
  }

  // Recommendation reason + persona suitability
  if (r.recommendation_reason || r.persona_suitability) {
    html += `<div class="rr-section">
      <div class="rr-section-title">Verdict</div>
      <div class="rr-text">${escHtml(r.recommendation_reason || '')}${r.persona_suitability ? ' ' + escHtml(r.persona_suitability) : ''}</div>
    </div>`;
  }

  // Navigation note
  if (r.navigation_note) {
    html += `<div class="rr-section">
      <div class="rr-section-title">Navigation</div>
      <div class="rr-text">${escHtml(r.navigation_note)}</div>
    </div>`;
  }

  body.innerHTML = html;
  overlay.classList.add('active');
}

function closeRouteReport() {
  document.getElementById('route-report-overlay').classList.remove('active');
}
```

**4. Wire up SSE handler** — in the `walkEventSource.onmessage` block, add:

```javascript
if (msg.type === 'route_report')  handleRouteReport(msg.data);
```

**5. Update `handleWalkComplete()`** to also show report if it came with the complete event:

```javascript
function handleWalkComplete(result) {
  // ... existing code ...
  
  // If route_report is embedded in complete event, show it
  if (result.route_report) {
    handleRouteReport(result.route_report);
  }
}
```

**6. Update `autoWalkReset()`** to close the report overlay:

```javascript
// Add at the beginning of autoWalkReset:
document.getElementById('route-report-overlay').classList.remove('active');
```

---

## Test Plan

After implementation, verify:

1. **Planner**: Start a walk → check logs for "Directions API: N waypoints" → check nav context is computed correctly → verify re-route triggers on deviation
2. **Reporter snapshots**: S2 trigger → check snapshot is generated before Interpreter runs → check Interpreter prompt includes the new snapshot
3. **Reporter episodes**: S2 trigger → check episode has narrative, significance_tag, persona_reaction fields
4. **Route report**: Walk completes → route_report SSE event arrives → overlay panel renders with all sections
5. **Fallback**: Temporarily break LLM connection → verify heuristic fallbacks work for both episode and route report
6. **Existing functionality**: Score chart, branch visualization, speech bubbles, headline still work as before

## Files Modified

- `src/agent/system2/planner.py` — Major rewrite (navigation logic)
- `src/agent/system2/reporter.py` — Major rewrite (3-phase design)
- `src/agent/orchestrator.py` — Remove navigation helpers, update autonomous_walk() S2 block and completion block
- `src/agent/memory/memory_manager.py` — Remove _generate_snapshot(), simplify update_with_system2_result(), simplify prepare_context_for_reasoning()
- `src/agent/system2/persona_reasoner.py` — Update reason() to use Reporter phases, track _last_snapshot_waypoint
- `src/agent/system2/models.py` — No changes needed (ReportResult dataclass can stay as-is for now)
- `demo/index.html` — Add route report overlay UI + SSE handler
- `demo/server.py` — Add route_report to complete event data

## Execution Order

1. Task 1 (Planner) first — it's a clean extraction with no dependencies on Reporter changes
2. Task 2 (Reporter) second — depends on Planner.get_summary() being available  
3. Task 3 (Frontend + SSE wiring) last — depends on route_report being generated

Within each task, edit existing files rather than creating new ones (except Planner is already a file, just needs rewriting).
