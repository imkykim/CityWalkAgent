"""Agent-specific logging helpers."""

from __future__ import annotations

from typing import Any, Optional

from src.utils.logging import StructuredLogger, get_logger


def get_agent_logger(agent_id: str) -> StructuredLogger:
    """Return a structured logger scoped to an agent id."""
    return get_logger(f"agent.{agent_id}")


class OrchestratorLogger:
    """Thin wrapper that keeps log formatting details out of orchestrator logic."""

    def __init__(self, logger: StructuredLogger) -> None:
        self._logger = logger

    def debug(self, message: str, **context: Any) -> None:
        self._logger.debug(message, **context)

    def info(self, message: str, **context: Any) -> None:
        self._logger.info(message, **context)

    def warning(self, message: str, **context: Any) -> None:
        self._logger.warning(message, **context)

    def error(self, message: str, **context: Any) -> None:
        self._logger.error(message, **context)

    def critical(self, message: str, **context: Any) -> None:
        self._logger.critical(message, **context)

    def phase_23_mode(self, skip_reasoning: bool) -> None:
        if skip_reasoning:
            self.debug(
                "Phase 2+3: Continuous analysis (reasoning SKIPPED — system1-only mode)"
            )
            return
        self.debug("Phase 2+3: Interleaved continuous analysis + memory + reasoning")

    def visual_change(self, waypoint_id: int, result: Any) -> None:
        self.debug(
            f"Visual change detection for waypoint {waypoint_id}",
            changed=getattr(result, "changed", None),
            distance=getattr(result, "phash_distance", None),
            reason=getattr(result, "reason", None),
        )

    def trigger_decision(
        self,
        waypoint_id: int,
        trigger_reason: Optional[Any],
        visual_change: bool,
        score_delta: float,
        distance_from_last: float,
    ) -> None:
        trigger = getattr(trigger_reason, "value", None) if trigger_reason else None
        self.debug(
            f"System 2 trigger decision for waypoint {waypoint_id}",
            trigger=trigger,
            visual_change=visual_change,
            score_delta=round(score_delta, 2),
            distance_from_last=distance_from_last,
        )

    def system2_waypoint(self, waypoint_id: int, trigger_reason: Any, result: Any) -> None:
        trigger = getattr(trigger_reason, "value", str(trigger_reason))
        significance = getattr(result, "significance", None)
        avoid_recommendation = getattr(result, "avoid_recommendation", None)
        interpretation = (getattr(result, "interpretation", "") or "")[:80]
        recommendation = getattr(result, "recommendation", None)

        self.info(
            f"WP {waypoint_id:>3}   S2 {trigger:<20}"
            f" significance={significance}"
            f" avoid={avoid_recommendation}"
            f" | {interpretation}"
        )
        self.info(f"WP {waypoint_id:>3}   S2 rec: {recommendation}")

    def image_saving_enabled(self, images_dir: Any) -> None:
        self.info(f"Image saving enabled → {images_dir}")

    def autonomous_walk_start(
        self, pano_id: str, dest_lat: float, dest_lng: float, max_steps: int
    ) -> None:
        self.info(
            f"autonomous_walk start | pano={pano_id[:12]} "
            f"dest=({dest_lat:.4f},{dest_lng:.4f}) max_steps={max_steps}"
        )

    def step_stuck(self, step: int, pano_id: str) -> None:
        self.warning(f"Step {step}: pano {pano_id[:12]} visited 5+ times — stuck, stopping")

    def step_metadata_error(self, step: int, error: Exception) -> None:
        self.warning(f"Step {step}: metadata error — {error}")

    def step_no_links(self, step: int) -> None:
        self.warning(f"Step {step}: no links — stopping")

    def step_arrived(self, step: int, dist_m: float) -> None:
        self.info(f"Step {step}: ARRIVED — dist={dist_m:.0f}m")

    def step_skip(self, step: int, heading_delta: float, dist_from_last: float) -> None:
        self.debug(
            f"Step {step}: SKIP (heading_Δ={heading_delta:.0f}°, dist={dist_from_last:.0f}m)"
        )

    def step_no_next_pano(self, step: int) -> None:
        self.warning(f"Step {step}: no next pano — stopping")

    def step_phash(self, step: int, phash_distance: Optional[float], visual_change: bool) -> None:
        self.debug(
            f"Step {step}: phash_distance={phash_distance} visual_change={visual_change}"
        )

    def step_image_saved(self, step: int, image_name: str) -> None:
        self.debug(f"Step {step}: image saved → {image_name}")

    def step_image_analysis_error(self, step: int, error: Exception) -> None:
        self.warning(f"Step {step}: image/analysis error — {error}")

    def step_intersection_trigger(self, step: int) -> None:
        self.debug(f"Step {step}: intersection trigger")

    def step_rerouting(self, step: int, nearest_wp_dist: float) -> None:
        self.info(f"Step {step}: route deviation {nearest_wp_dist:.0f}m — re-routing")

    def step_branch_choice(
        self,
        step: int,
        is_intersection: bool,
        chosen_direction: str,
        chosen_heading: float,
        confidence: float,
    ) -> None:
        reason_label = "intersection" if is_intersection else "S2 rethink"
        self.info(
            f"Step {step:>3} | {reason_label} → {chosen_direction} "
            f"({chosen_heading:.0f}°) conf={confidence:.2f}"
        )

    def step_branch_logged(self, step: int, is_intersection: bool, episodes: int) -> None:
        significance = "high" if is_intersection else "medium"
        self.debug(
            f"Step {step}: branch result logged to LTM episodes"
            f" | significance={significance}"
            f" | episodes={episodes}"
        )

    def step_branch_error(self, step: int, error: Exception) -> None:
        self.warning(f"Step {step}: branch error — {error}")

    def step_initial_heading(self, step: int, chosen_heading: float) -> None:
        self.debug(f"Step {step}: initial heading → {chosen_heading:.0f}° (toward dest)")

    def step_lookahead_jump(self, step: int, next_pano_id: str, lookahead_depth: int) -> None:
        self.info(
            f"Step {step}: jump to lookahead end pano {next_pano_id[:12]} "
            f"(depth={lookahead_depth})"
        )

    def step_lookahead_jump_skipped(
        self, step: int, branch_last_pano: Optional[str], current_pano_id: str
    ) -> None:
        self.info(
            f"Step {step}: branch lookahead jump skipped "
            f"(last_pano={branch_last_pano!r}, current={current_pano_id[:12]}) → using link"
        )

    def step_progress(
        self,
        step: int,
        dist_m: float,
        cardinal: str,
        dest_bearing: float,
        chosen_heading: float,
        next_pano_id: str,
    ) -> None:
        self.info(
            f"Step {step:>3} | dist={dist_m:.0f}m {cardinal}({dest_bearing:.0f}°) "
            f"→ heading={chosen_heading:.0f}° | next={next_pano_id[:12]}"
        )

    def route_report_generated(self, recommendation: Any) -> None:
        self.info(f"Route report generated: {recommendation}")

    def route_report_failed(self, error: Exception) -> None:
        self.warning(f"Route report generation failed: {error}")

    def walk_outputs_saved(self, output_dir: Any, files: Any) -> None:
        self.info(f"Walk outputs saved → {output_dir}", files=files)

    def walk_complete(
        self,
        arrived: bool,
        steps: int,
        analyzed_step_count: int,
        total_steps: int,
        final_dist: float,
    ) -> None:
        self.info(
            f"Walk complete | arrived={arrived} steps={steps} "
            f"analyzed={analyzed_step_count}/{total_steps} steps "
            f"final_dist={final_dist:.0f}m"
        )
