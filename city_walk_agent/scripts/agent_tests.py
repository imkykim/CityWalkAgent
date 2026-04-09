#!/usr/bin/env python3
"""
CityWalkAgent — Automated Test Runner

Runs a matrix of autonomous walk experiments and generates a cross-experiment
comparison report. Designed for systematic validation of:
  - Route selection quality (arrival rate, path efficiency)
  - S1 score consistency across environments
  - S2 trigger appropriateness (rate, trigger reasons)
  - Persona differentiation (same route, different persona)
  - Branch decision quality (destination awareness, score sensitivity)

Usage:
    # Full matrix (all routes × all personas)
    python scripts/agent_tests.py --matrix full

    # Quick sanity check (1 route × 1 persona, 10 steps)
    python scripts/agent_tests.py --matrix quick

    # Single experiment
    python scripts/agent_tests.py --matrix single \\
        --start 1.2835,103.858 --dest 1.2900,103.852 \\
        --persona homebuyer --max-steps 40

    # Custom matrix from JSON config
    python scripts/agent_tests.py --matrix custom --config tests/test_matrix.json

Output:
    outputs/test_runs/<run_id>/
    ├── experiment_matrix.json        # what was planned
    ├── <persona>_<timestamp>/        # per-experiment outputs
    │   ├── run_metadata.json
    │   ├── walk_log.json
    │   ├── score_timeline.csv
    │   ├── branch_decisions.json
    │   ├── s2_reasoning.json
    │   ├── route_map.html
    │   ├── score_chart.png
    │   └── summary_radar.png
    └── comparison_report.json        # cross-experiment comparison
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Project imports ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
for p in (PROJECT_ROOT, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.core import settings
from src.agent.orchestrator import CityWalkAgent
from src.agent.config.personalities import get_preset, list_presets
from src.utils.logging import get_logger, set_global_log_level

logger = get_logger("test_runner")


def _first_key(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return first non-None value for keys from dict-like object."""
    if not isinstance(d, dict):
        return default
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ============================================================================
# Test Routes — curated start/dest pairs for different urban environments
# ============================================================================

TEST_ROUTES: Dict[str, Dict[str, Any]] = {
    "sg_marina_promenade": {
        "name": "Marina Bay Promenade",
        "start": (1.2812, 103.8584),
        "dest": (1.2838, 103.8562),
        "description": "Waterfront boardwalk, wide & scenic, ~350m",
        "expected_character": "high_quality_scenic",
    },
    "sg_tiong_bahru": {
        "name": "Tiong Bahru Residential",
        "start": (1.2860, 103.8310),
        "dest": (1.2840, 103.8338),
        "description": "Quiet art-deco residential, cafes, ~350m",
        "expected_character": "residential_charming",
    },
    "sg_chinatown_heritage": {
        "name": "Chinatown Heritage",
        "start": (1.2827, 103.8470),
        "dest": (1.2805, 103.8450),
        "description": "South Bridge Road shophouses, temples, ~300m",
        "expected_character": "dense_cultural",
    },
    "sg_orchard_road": {
        "name": "Orchard Road Commercial",
        "start": (1.3025, 103.8365),
        "dest": (1.3040, 103.8330),
        "description": "Major shopping boulevard, busy & commercial, ~350m",
        "expected_character": "commercial_busy",
    },
    "sg_kampong_glam": {
        "name": "Kampong Glam / Arab St",
        "start": (1.3015, 103.8590),
        "dest": (1.3003, 103.8618),
        "description": "Cultural district, street art, textiles, ~350m",
        "expected_character": "cultural_vibrant",
    },
    "sg_toa_payoh_hdb": {
        "name": "Toa Payoh HDB Estate",
        "start": (1.3340, 103.8490),
        "dest": (1.3320, 103.8510),
        "description": "Public housing estate, covered walks, playgrounds, ~300m",
        "expected_character": "residential_utilitarian",
    },
    "hk_sham_shui_po": {
        "name": "HK Sham Shui Po Markets",
        "start": (22.3307, 114.1625),
        "dest": (22.3288, 114.1643),
        "description": "Local market alleys, textiles & toys, ~300m",
        "expected_character": "gritty_authentic",
    },
    "hk_tst_waterfront": {
        "name": "HK TST Waterfront",
        "start": (22.2935, 114.1720),
        "dest": (22.2950, 114.1745),
        "description": "Victoria Harbour promenade, no cars, ~300m",
        "expected_character": "high_quality_scenic",
    },
    "hk_mong_kok": {
        "name": "HK Mong Kok Market Streets",
        "start": (22.3193, 114.1694),  # Tung Choi St (Ladies Market) north end
        "dest": (22.3150, 114.1700),   # towards Argyle St south
        "description": "Pedestrian market streets, dense grid, ~500m",
        "expected_character": "hyper_dense_commercial",
    },
    "hk_causeway_bay": {
        "name": "HK Causeway Bay Luxury District",
        "start": (22.2785, 114.1840),    # Pak Sha Road, Stussy 근처
        "dest": (22.2800, 114.1855),     # Hysan Place (Apple Store) 근처
        "description": "Lee Gardens luxury district × Hysan Place, ~200m",
        "expected_character": "high_wealthy_commercial",
    },
    "hk_sheung_wan": {
        "name": "HK Sheung Wan Heritage",
        "start": (22.2845, 114.1500),
        "dest": (22.2860, 114.1520),
        "description": "Hilly pedestrian heritage, temples, galleries, ~300m",
        "expected_character": "cultural_hilly",
    },
}

# ============================================================================
# Test Matrices — predefined experiment configurations
# ============================================================================

MATRICES = {
    "quick": {
        "routes": ["hk_tst_waterfront"],
        "personas": ["homebuyer"],
        "max_steps": 10,
        "lookahead_depth": 1,
        "save_images": True,
    },
    "causeway_bay": {
        "routes": ["hk_causeway_bay"],
        "personas": [
            "homebuyer",
            "parent_with_kids",
            "photographer",
            "runner",
            "tourist",
        ],
        "max_steps": 30,
        "lookahead_depth": 3,
        "save_images": True,
    },
    "mong_kok": {
        "routes": ["hk_mong_kok"],
        "personas": [
            "homebuyer",
            "parent_with_kids",
            "photographer",
            "runner",
            "tourist",
        ],
        "max_steps": 30,
        "lookahead_depth": 3,
        "save_images": True,
    },
    "persona_compare": {
        "routes": ["sg_toa_payoh_hdb", "hk_mong_kok"],
        "personas": [
            "homebuyer",
            "parent_with_kids",
            "photographer",
            "runner",
        ],
        "max_steps": 30,
        "lookahead_depth": 1,
        "save_images": True,
    },
    "route_compare": {
        "routes": [
            "sg_marina_promenade",
            "sg_tiong_bahru",
            "sg_chinatown_heritage",
            "sg_orchard_road",
            "sg_kampong_glam",
            "sg_toa_payoh_hdb",
        ],
        "personas": ["homebuyer"],
        "max_steps": 20,
        "lookahead_depth": 1,
        "save_images": True,
    },
    "full": {
        "routes": [
            "sg_marina_promenade",
            "sg_chinatown_heritage",
            "sg_toa_payoh_hdb",
        ],
        "personas": ["homebuyer", "parent_with_kids", "photographer", "tourist"],
        "max_steps": 20,
        "lookahead_depth": 1,
        "save_images": True,
    },
}


# ============================================================================
# Experiment Data Classes
# ============================================================================


@dataclass
class ExperimentSpec:
    """Single experiment specification."""

    route_id: str
    route_name: str
    start: Tuple[float, float]
    dest: Tuple[float, float]
    persona: str
    max_steps: int
    lookahead_depth: int
    save_images: bool = False


@dataclass
class ExperimentResult:
    """Result of a single experiment."""

    spec: ExperimentSpec
    arrived: bool
    analyzed_steps: int
    total_steps: int
    skipped_steps: int
    skip_rate: float
    final_distance_m: float
    duration_seconds: float
    output_dir: str
    avg_score: Optional[float] = None
    score_std: Optional[float] = None
    dim_means: Dict[str, float] = field(default_factory=dict)
    branch_count: int = 0
    s2_trigger_count: int = 0
    s2_trigger_rate_of_analyzed: float = 0.0
    s2_trigger_rate_of_total: float = 0.0
    s2_trigger_rate: float = 0.0
    gating_signal_counts: Dict[str, int] = field(default_factory=dict)
    reroute_count: int = 0
    urgency_tier_counts: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None


# ============================================================================
# Test Runner
# ============================================================================


class TestRunner:
    """Execute a matrix of walk experiments and generate comparison reports."""

    def __init__(
        self,
        output_base: Path = Path("outputs/test_runs"),
        framework_id: str = "place_pulse_2.0",
        log_level: str = "INFO",
    ):
        self.output_base = output_base
        self.framework_id = framework_id
        self.log_level = log_level
        set_global_log_level(log_level)

    def build_matrix(self, matrix_config: Dict[str, Any]) -> List[ExperimentSpec]:
        """Build experiment list from matrix configuration."""
        specs = []
        routes = matrix_config.get("routes", [])
        personas = matrix_config.get("personas", [])
        max_steps = matrix_config.get("max_steps", 40)
        lookahead = matrix_config.get("lookahead_depth", 1)
        save_images = matrix_config.get("save_images", False)

        for route_id in routes:
            if route_id not in TEST_ROUTES:
                logger.warning(f"Unknown route: {route_id}, skipping")
                continue
            route = TEST_ROUTES[route_id]
            for persona in personas:
                specs.append(
                    ExperimentSpec(
                        route_id=route_id,
                        route_name=route["name"],
                        start=route["start"],
                        dest=route["dest"],
                        persona=persona,
                        max_steps=max_steps,
                        lookahead_depth=lookahead,
                        save_images=save_images,
                    )
                )
        return specs

    def _dump_schema_diagnosis(
        self,
        result: Dict[str, Any],
        schema_dump_path: Optional[Path],
    ) -> None:
        """Persist raw autonomous_walk schema for drift diagnostics."""
        if not schema_dump_path:
            return

        route_taken = _first_key(
            result,
            "route_taken",
            "steps",
            "walk_steps",
            default=[],
        )
        if not isinstance(route_taken, list):
            route_taken = []

        sample_step = route_taken[0] if route_taken else {}
        if not isinstance(sample_step, dict):
            sample_step = {}

        score_block = _first_key(
            sample_step,
            "scores",
            "dim_scores",
            "dimension_scores",
            default={},
        )
        if not isinstance(score_block, dict):
            score_block = {}

        diagnosis = {
            "captured_at": datetime.now().isoformat(),
            "top_level_keys": sorted(list(result.keys())) if isinstance(result, dict) else [],
            "top_level_sample": {
                "arrived": _first_key(result, "arrived"),
                "steps": _first_key(result, "steps"),
                "analyzed_steps": _first_key(result, "analyzed_steps"),
                "total_steps": _first_key(result, "total_steps"),
                "skip_rate": _first_key(result, "skip_rate"),
                "reroute_count": _first_key(result, "reroute_count"),
            },
            "route_taken_len": len(route_taken),
            "sample_step_keys": sorted(list(sample_step.keys())),
            "sample_step": sample_step,
            "score_dict_keys": sorted(list(score_block.keys())),
            "planner_summary_keys": sorted(
                list((_first_key(result, "planner_summary", default={}) or {}).keys())
            ),
            "route_report_keys": sorted(
                list((_first_key(result, "route_report", default={}) or {}).keys())
            ),
        }

        schema_dump_path.write_text(
            json.dumps(diagnosis, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Schema diagnosis saved: {schema_dump_path}")

    async def run_single(
        self,
        spec: ExperimentSpec,
        experiment_dir: Path,
        schema_dump_path: Optional[Path] = None,
    ) -> ExperimentResult:
        """Run a single autonomous walk experiment."""
        logger.info(
            f"Starting experiment: {spec.route_name} × {spec.persona} "
            f"(max {spec.max_steps} steps)"
        )

        start_time = time.time()
        try:
            personality = get_preset(spec.persona, self.framework_id)
            agent = CityWalkAgent(
                agent_id=f"test_{spec.persona}_{int(time.time())}",
                personality=personality,
                framework_id=self.framework_id,
            )

            result = await agent.autonomous_walk(
                start_lat=spec.start[0],
                start_lng=spec.start[1],
                dest_lat=spec.dest[0],
                dest_lng=spec.dest[1],
                max_steps=spec.max_steps,
                output_dir=experiment_dir,
                save_images=spec.save_images,
                lookahead_depth=spec.lookahead_depth,
            )

            duration = time.time() - start_time
            self._dump_schema_diagnosis(result, schema_dump_path)

            # Extract summary stats from result
            route_taken = _first_key(
                result,
                "route_taken",
                "steps_taken",
                "walk_steps",
                default=[],
            )
            if not isinstance(route_taken, list):
                route_taken = []

            analyzed_steps = _to_int(
                _first_key(result, "analyzed_steps", "analysis_steps", default=len(route_taken)),
                default=len(route_taken),
            )
            total_steps = _to_int(
                _first_key(result, "total_steps", "steps", default=max(analyzed_steps, len(route_taken))),
                default=max(analyzed_steps, len(route_taken)),
            )
            if total_steps < analyzed_steps:
                total_steps = analyzed_steps

            skipped_steps = _to_int(
                _first_key(result, "skipped_steps", "skip_steps", default=max(total_steps - analyzed_steps, 0)),
                default=max(total_steps - analyzed_steps, 0),
            )
            skip_rate_raw = _first_key(result, "skip_rate", default=None)
            skip_rate = (
                _to_float(skip_rate_raw, default=0.0)
                if skip_rate_raw is not None
                else (skipped_steps / max(total_steps, 1))
            )

            planner_summary = _first_key(result, "planner_summary", default={}) or {}
            if not isinstance(planner_summary, dict):
                planner_summary = {}
            reroute_count = _to_int(
                _first_key(result, "reroute_count", default=_first_key(planner_summary, "reroute_count", default=0)),
                default=0,
            )

            raw_gating_counts = _first_key(result, "gating_signal_counts", default=None)
            has_top_level_gating = isinstance(raw_gating_counts, dict)
            gating_signal_counts = raw_gating_counts or {}
            if not isinstance(gating_signal_counts, dict):
                gating_signal_counts = {}
            gating_signal_counts = {
                "heading": _to_int(gating_signal_counts.get("heading", 0)),
                "distance": _to_int(gating_signal_counts.get("distance", 0)),
                "phash": _to_int(gating_signal_counts.get("phash", 0)),
                "intersection": _to_int(gating_signal_counts.get("intersection", 0)),
            }

            raw_urgency_counts = _first_key(result, "urgency_tier_counts", default=None)
            has_top_level_urgency = isinstance(raw_urgency_counts, dict)
            urgency_tier_counts = raw_urgency_counts or {}
            if not isinstance(urgency_tier_counts, dict):
                urgency_tier_counts = {}
            urgency_tier_counts = {
                "explore": _to_int(urgency_tier_counts.get("explore", 0)),
                "navigate": _to_int(urgency_tier_counts.get("navigate", 0)),
                "converge": _to_int(urgency_tier_counts.get("converge", 0)),
            }

            all_avgs = []
            dim_all: Dict[str, List[float]] = {}
            branch_count = 0
            s2_count = 0

            for step in route_taken:
                if not isinstance(step, dict):
                    continue

                analyzed = _first_key(step, "analyzed", default=None)
                if analyzed is None:
                    analyzed = True

                scores = _first_key(
                    step,
                    "scores",
                    "dim_scores",
                    "dimension_scores",
                    default={},
                )
                if not isinstance(scores, dict):
                    scores = {}

                vals = [v for v in scores.values() if isinstance(v, (int, float))]
                if analyzed and vals:
                    all_avgs.append(sum(vals) / len(vals))
                for d, v in scores.items():
                    if analyzed and isinstance(v, (int, float)):
                        dim_all.setdefault(d, []).append(v)
                if _first_key(step, "branch_triggered", default=False):
                    branch_count += 1

                s2_triggered = _first_key(step, "s2_triggered", default=None)
                if s2_triggered is None:
                    s2_triggered = bool(
                        _first_key(step, "trigger_reason", "s2_trigger_reason", default=None)
                    )
                if s2_triggered:
                    s2_count += 1

                step_tier = _first_key(step, "urgency_tier", default=None)
                if (not has_top_level_urgency) and isinstance(step_tier, str):
                    urgency_tier_counts[step_tier] = urgency_tier_counts.get(step_tier, 0) + 1

                step_signals = _first_key(step, "gating_signals", "trigger_signals", default=[])
                if isinstance(step_signals, dict):
                    step_signals = [k for k, v in step_signals.items() if v]
                if not isinstance(step_signals, list):
                    step_signals = []
                if not has_top_level_gating:
                    for signal in step_signals:
                        key = str(signal).strip().lower()
                        if key in gating_signal_counts:
                            gating_signal_counts[key] += 1

            dim_means = {
                d: round(sum(vs) / len(vs), 3) for d, vs in dim_all.items() if vs
            }
            score_std = (
                round(statistics.pstdev(all_avgs), 3) if len(all_avgs) >= 2 else None
            )

            s2_rate_of_analyzed = (
                round(s2_count / max(analyzed_steps, 1), 3)
                if analyzed_steps > 0
                else 0.0
            )
            s2_rate_of_total = (
                round(s2_count / max(total_steps, 1), 3)
                if total_steps > 0
                else 0.0
            )

            return ExperimentResult(
                spec=spec,
                arrived=bool(_first_key(result, "arrived", default=False)),
                analyzed_steps=analyzed_steps,
                total_steps=total_steps,
                skipped_steps=skipped_steps,
                skip_rate=round(skip_rate, 3),
                final_distance_m=_to_float(_first_key(result, "final_distance_m", default=-1), default=-1),
                duration_seconds=round(duration, 1),
                output_dir=str(experiment_dir),
                avg_score=round(sum(all_avgs) / len(all_avgs), 3) if all_avgs else None,
                score_std=score_std,
                dim_means=dim_means,
                branch_count=branch_count,
                s2_trigger_count=s2_count,
                s2_trigger_rate_of_analyzed=s2_rate_of_analyzed,
                s2_trigger_rate_of_total=s2_rate_of_total,
                s2_trigger_rate=s2_rate_of_total,  # backward-compatible legacy metric
                gating_signal_counts=gating_signal_counts,
                reroute_count=reroute_count,
                urgency_tier_counts=urgency_tier_counts,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Experiment failed: {e}")
            import traceback

            traceback.print_exc()
            return ExperimentResult(
                spec=spec,
                arrived=False,
                analyzed_steps=0,
                total_steps=0,
                skipped_steps=0,
                skip_rate=0.0,
                final_distance_m=-1,
                duration_seconds=round(duration, 1),
                output_dir=str(experiment_dir),
                error=str(e),
            )

    async def run_matrix(
        self,
        specs: List[ExperimentSpec],
        diagnose: bool = False,
    ) -> List[ExperimentResult]:
        """Run all experiments sequentially (not parallel — VLM rate limits)."""
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_base / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        schema_dump_path = run_dir / "schema_diagnosis.json" if diagnose else None

        # Save experiment plan
        plan = {
            "run_id": run_id,
            "total_experiments": len(specs),
            "started_at": datetime.now().isoformat(),
            "experiments": [asdict(s) for s in specs],
        }
        with open(run_dir / "experiment_matrix.json", "w") as f:
            json.dump(plan, f, indent=2, default=str)

        results: List[ExperimentResult] = []
        for i, spec in enumerate(specs):
            print(f"\n{'='*60}")
            print(f"  Experiment {i+1}/{len(specs)}")
            print(f"  Route:   {spec.route_name}")
            print(f"  Persona: {spec.persona}")
            print(f"  Steps:   max {spec.max_steps}")
            print(f"{'='*60}\n")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_dir = run_dir / f"{spec.route_id}_{spec.persona}_{timestamp}"
            exp_result = await self.run_single(
                spec,
                exp_dir,
                schema_dump_path=schema_dump_path if (diagnose and i == 0) else None,
            )
            results.append(exp_result)

            # Print quick summary
            status = (
                "✅ ARRIVED"
                if exp_result.arrived
                else (
                    f"❌ FAILED: {exp_result.error}"
                    if exp_result.error
                    else "⏹ MAX STEPS"
                )
            )
            print(f"\n  {status}")
            print(
                f"  Steps: {exp_result.analyzed_steps}/{exp_result.total_steps} "
                f"(skip {exp_result.skip_rate:.0%}) | "
                f"reroute {exp_result.reroute_count} | "
                f"Time: {exp_result.duration_seconds}s | "
                f"Avg score: {exp_result.avg_score}"
            )
            print(
                f"  Branches: {exp_result.branch_count} | "
                f"S2 triggers: {exp_result.s2_trigger_count} "
                f"(analyzed {exp_result.s2_trigger_rate_of_analyzed:.0%})"
            )

        # Generate comparison report
        report = self._generate_comparison_report(results, run_id)
        report_path = run_dir / "comparison_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Print summary table
        self._print_summary_table(results)

        print(f"\n📁 All outputs saved to: {run_dir}")
        print(f"📊 Comparison report:    {report_path}")
        if diagnose:
            print(f"🧭 Schema diagnosis:    {schema_dump_path}")

        return results

    def _generate_comparison_report(
        self,
        results: List[ExperimentResult],
        run_id: str,
    ) -> Dict[str, Any]:
        """Generate cross-experiment comparison analysis."""
        report: Dict[str, Any] = {
            "run_id": run_id,
            "generated_at": datetime.now().isoformat(),
            "total_experiments": len(results),
            "successful": sum(1 for r in results if not r.error),
            "failed": sum(1 for r in results if r.error),
        }

        successful = [r for r in results if not r.error]
        if not successful:
            report["summary"] = "All experiments failed"
            return report

        # ── Overall statistics ───────────────────────────────────────────
        in_target_band = [
            r for r in successful if 0.15 <= r.s2_trigger_rate_of_analyzed <= 0.25
        ]
        report["overall"] = {
            "arrival_rate": round(
                sum(1 for r in successful if r.arrived) / len(successful), 3
            ),
            "avg_analyzed_steps": round(
                sum(r.analyzed_steps for r in successful) / len(successful), 1
            ),
            "avg_total_steps": round(
                sum(r.total_steps for r in successful) / len(successful), 1
            ),
            "avg_skip_rate": round(
                sum(r.skip_rate for r in successful) / len(successful), 3
            ),
            "avg_duration_s": round(
                sum(r.duration_seconds for r in successful) / len(successful), 1
            ),
            "avg_reroute_count": round(
                sum(r.reroute_count for r in successful) / len(successful), 3
            ),
            "avg_score": (
                round(
                    sum(r.avg_score for r in successful if r.avg_score)
                    / sum(1 for r in successful if r.avg_score),
                    3,
                )
                if any(r.avg_score for r in successful)
                else None
            ),
            "avg_s2_trigger_rate": round(
                sum(r.s2_trigger_rate for r in successful) / len(successful), 3
            ),
            "avg_s2_rate_of_analyzed": round(
                sum(r.s2_trigger_rate_of_analyzed for r in successful) / len(successful),
                3,
            ),
            "avg_s2_rate_of_total": round(
                sum(r.s2_trigger_rate_of_total for r in successful) / len(successful),
                3,
            ),
            "s2_rate_in_target_band": round(
                len(in_target_band) / len(successful),
                3,
            ),
            "avg_score_std_within_route": (
                round(
                    sum(r.score_std for r in successful if r.score_std is not None)
                    / max(1, sum(1 for r in successful if r.score_std is not None)),
                    3,
                )
                if any(r.score_std is not None for r in successful)
                else None
            ),
        }

        # ── Per-route comparison ─────────────────────────────────────────
        routes_seen = set(r.spec.route_id for r in successful)
        route_comparison = {}
        for route_id in sorted(routes_seen):
            route_results = [r for r in successful if r.spec.route_id == route_id]
            persona_scores = [r.avg_score for r in route_results if r.avg_score is not None]
            persona_avg_score_gap = (
                round(max(persona_scores) - min(persona_scores), 3)
                if len(persona_scores) >= 2
                else 0.0
            )

            dim_by_persona: Dict[str, List[float]] = {}
            for rr in route_results:
                for dim, val in rr.dim_means.items():
                    dim_by_persona.setdefault(dim, []).append(val)

            per_dim_std = {
                dim: round(statistics.pstdev(vals), 3)
                for dim, vals in dim_by_persona.items()
                if len(vals) >= 2
            }
            per_dim_gap = {
                dim: round(max(vals) - min(vals), 3)
                for dim, vals in dim_by_persona.items()
                if len(vals) >= 2
            }
            route_comparison[route_id] = {
                "route_name": route_results[0].spec.route_name,
                "experiments": len(route_results),
                "arrival_rate": round(
                    sum(1 for r in route_results if r.arrived) / len(route_results), 3
                ),
                "avg_score": round(
                    sum(r.avg_score for r in route_results if r.avg_score)
                    / max(1, sum(1 for r in route_results if r.avg_score)),
                    3,
                ),
                "personas_tested": [r.spec.persona for r in route_results],
                "persona_avg_score_gap": persona_avg_score_gap,
                "persona_gap_in_target_band": (
                    3.0 <= persona_avg_score_gap <= 6.0
                    if len(persona_scores) >= 2
                    else False
                ),
                "persona_divergence": {
                    "per_dimension_std": per_dim_std,
                    "per_dimension_gap": per_dim_gap,
                    "max_dimension_gap": (max(per_dim_gap.values()) if per_dim_gap else 0.0),
                    "mean_dimension_std": (
                        round(sum(per_dim_std.values()) / len(per_dim_std), 3)
                        if per_dim_std
                        else 0.0
                    ),
                },
            }
        report["by_route"] = route_comparison

        # ── Per-persona comparison ───────────────────────────────────────
        personas_seen = set(r.spec.persona for r in successful)
        persona_comparison = {}
        for persona in sorted(personas_seen):
            persona_results = [r for r in successful if r.spec.persona == persona]
            dim_agg: Dict[str, List[float]] = {}
            for r in persona_results:
                for d, v in r.dim_means.items():
                    dim_agg.setdefault(d, []).append(v)

            persona_comparison[persona] = {
                "experiments": len(persona_results),
                "arrival_rate": round(
                    sum(1 for r in persona_results if r.arrived) / len(persona_results),
                    3,
                ),
                "avg_score": round(
                    sum(r.avg_score for r in persona_results if r.avg_score)
                    / max(1, sum(1 for r in persona_results if r.avg_score)),
                    3,
                ),
                "avg_dim_scores": {
                    d: round(sum(vs) / len(vs), 3) for d, vs in dim_agg.items() if vs
                },
                "avg_branch_count": round(
                    sum(r.branch_count for r in persona_results) / len(persona_results),
                    1,
                ),
                "avg_skip_rate": round(
                    sum(r.skip_rate for r in persona_results) / len(persona_results),
                    3,
                ),
                "avg_s2_rate": round(
                    sum(r.s2_trigger_rate for r in persona_results)
                    / len(persona_results),
                    3,
                ),
                "avg_s2_rate_of_analyzed": round(
                    sum(r.s2_trigger_rate_of_analyzed for r in persona_results)
                    / len(persona_results),
                    3,
                ),
                "avg_reroute_count": round(
                    sum(r.reroute_count for r in persona_results) / len(persona_results),
                    3,
                ),
            }
        report["by_persona"] = persona_comparison

        gating_signal_totals = {"heading": 0, "distance": 0, "phash": 0, "intersection": 0}
        urgency_tier_totals = {"explore": 0, "navigate": 0, "converge": 0}
        for r in successful:
            for k, v in (r.gating_signal_counts or {}).items():
                gating_signal_totals[k] = gating_signal_totals.get(k, 0) + _to_int(v, 0)
            for k, v in (r.urgency_tier_counts or {}).items():
                urgency_tier_totals[k] = urgency_tier_totals.get(k, 0) + _to_int(v, 0)

        report["gating_signal_totals"] = gating_signal_totals
        report["urgency_tier_totals"] = urgency_tier_totals

        # ── Experiment details ───────────────────────────────────────────
        report["experiments"] = []
        for r in results:
            report["experiments"].append(
                {
                    "route": r.spec.route_id,
                    "persona": r.spec.persona,
                    "arrived": r.arrived,
                    "analyzed_steps": r.analyzed_steps,
                    "total_steps": r.total_steps,
                    "skipped_steps": r.skipped_steps,
                    "skip_rate": r.skip_rate,
                    "final_distance_m": r.final_distance_m,
                    "duration_s": r.duration_seconds,
                    "avg_score": r.avg_score,
                    "score_std": r.score_std,
                    "dim_means": r.dim_means,
                    "branches": r.branch_count,
                    "s2_triggers": r.s2_trigger_count,
                    "s2_rate": r.s2_trigger_rate,
                    "s2_rate_of_analyzed": r.s2_trigger_rate_of_analyzed,
                    "s2_rate_of_total": r.s2_trigger_rate_of_total,
                    "gating_signal_counts": r.gating_signal_counts,
                    "reroute_count": r.reroute_count,
                    "urgency_tier_counts": r.urgency_tier_counts,
                    "output_dir": r.output_dir,
                    "error": r.error,
                }
            )

        return report

    def _print_summary_table(self, results: List[ExperimentResult]):
        """Print a concise summary table to terminal."""
        print(f"\n{'='*116}")
        print("  EXPERIMENT SUMMARY")
        print(f"{'='*116}")
        print(
            f"{'Route':<20} {'Persona':<15} {'A/T':>7} {'Skip':>6} {'Std':>5} "
            f"{'Rrt':>4} {'Dist':>7} {'Avg':>5} {'Branch':>6} {'S2A%':>6} {'Time':>6} {'Status'}"
        )
        print(f"{'-'*116}")

        for r in results:
            if r.error:
                status = "❌ ERR"
            elif r.arrived:
                status = "✅"
            else:
                status = "⏹"

            print(
                f"{r.spec.route_id:<20} {r.spec.persona:<15} "
                f"{f'{r.analyzed_steps}/{r.total_steps}':>7} {r.skip_rate:>5.0%} "
                f"{(r.score_std if r.score_std is not None else 0):>5.2f} "
                f"{r.reroute_count:>4} "
                f"{r.final_distance_m:>7.0f}m {r.avg_score or 0:>5.1f} {r.branch_count:>6} "
                f"{r.s2_trigger_rate_of_analyzed:>5.0%} {r.duration_seconds:>5.0f}s "
                f"{status}"
            )

        print(f"{'='*116}")


# ============================================================================
# CLI Entry Point
# ============================================================================


def parse_coord(value: str) -> Tuple[float, float]:
    lat, lon = value.split(",")
    return float(lat.strip()), float(lon.strip())


def main():
    parser = argparse.ArgumentParser(
        description="CityWalkAgent — Automated Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Predefined matrices:
  quick           1 route × 1 persona × 10 steps (~1-2 min)
  persona_compare 2 routes × 4 personas × 30 steps
  route_compare   6 routes × 1 persona × 20 steps
  full            3 routes × 4 personas × 20 steps
        """,
    )
    parser.add_argument(
        "--matrix",
        choices=list(MATRICES.keys()) + ["single", "custom"],
        default="quick",
        help="Predefined test matrix or 'single' for one experiment",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Custom matrix JSON config file (required with --matrix custom)",
    )
    parser.add_argument(
        "--start", type=parse_coord, help="Start coords for single mode"
    )
    parser.add_argument("--dest", type=parse_coord, help="Dest coords for single mode")
    parser.add_argument(
        "--persona", default="homebuyer", help="Persona for single mode"
    )
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--lookahead-depth", type=int, default=1)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Dump raw autonomous_walk schema from first experiment to schema_diagnosis.json",
    )
    parser.add_argument("--output-base", type=Path, default=Path("outputs/test_runs"))
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        type=str.upper,
    )

    args = parser.parse_args()
    runner = TestRunner(
        output_base=args.output_base,
        log_level=args.log_level,
    )

    if args.matrix == "single":
        # Single experiment
        start = args.start or (1.2835, 103.858)
        dest = args.dest or (1.2900, 103.852)
        specs = [
            ExperimentSpec(
                route_id="custom",
                route_name="Custom Route",
                start=start,
                dest=dest,
                persona=args.persona,
                max_steps=args.max_steps,
                lookahead_depth=args.lookahead_depth,
                save_images=args.save_images,
            )
        ]
    elif args.matrix == "custom":
        if not args.config:
            parser.error("--matrix custom requires --config <path_to_json>")
        # Custom matrix from JSON
        with open(args.config) as f:
            custom_config = json.load(f)
        # Allow custom routes in config
        if "routes_def" in custom_config:
            TEST_ROUTES.update(custom_config["routes_def"])
        specs = runner.build_matrix(custom_config)
    else:
        # Predefined matrix
        matrix_config = MATRICES[args.matrix].copy()
        if args.max_steps != 40:
            matrix_config["max_steps"] = args.max_steps
        matrix_config["lookahead_depth"] = args.lookahead_depth
        matrix_config["save_images"] = args.save_images
        specs = runner.build_matrix(matrix_config)

    if not specs:
        print("❌ No experiments to run. Check your configuration.")
        sys.exit(1)

    print(f"\n🧪 CityWalkAgent Test Runner")
    print(f"   Experiments: {len(specs)}")
    print(f"   Output:      {args.output_base}")
    for i, s in enumerate(specs):
        print(f"   [{i+1}] {s.route_name} × {s.persona} (max {s.max_steps} steps)")
    print()

    asyncio.run(runner.run_matrix(specs, diagnose=args.diagnose))


if __name__ == "__main__":
    main()
